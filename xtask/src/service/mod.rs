mod cache_manager;
mod error;
mod response;

use crate::BaseArgs;
use cache_manager::CacheManager;
use error::*;
use http_body_util::{BodyExt, combinators::BoxBody};
use hyper::{
    Method, Request, Response,
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
};
use hyper_util::rt::TokioIo;
use llama_cu::{Message, Received, ReturnReason, Service, SessionId, Terminal, TextBuf, utok};
use log::{info, warn};
use openai_struct::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use response::{error, text_stream};
use std::{
    collections::BTreeMap,
    ffi::c_int,
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
};
use tokio::{
    net::TcpListener,
    sync::mpsc::{self, UnboundedSender},
};
use tokio_stream::wrappers::UnboundedReceiverStream;

const V1_CHAT_COMPLETIONS: &str = "/v1/chat/completions";

#[derive(Args)]
pub struct ServiceArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(short, long)]
    port: u16,
}

impl ServiceArgs {
    pub fn service(self) {
        let Self { base, port } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(start_infer_service(
                base.model,
                port,
                gpus,
                max_steps,
                !base.no_cuda_graph,
            ))
            .unwrap()
    }
}

async fn start_infer_service(
    model: PathBuf,
    port: u16,
    gpus: Box<[c_int]>,
    max_steps: usize,
    use_cuda_graph: bool,
) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let service = Service::new(model, &gpus, use_cuda_graph);
    let sessions: BTreeMap<SessionId, SessionInfo> = BTreeMap::new();

    let service_manager = Arc::new(ServiceManager {
        terminal: service.terminal().clone(),
        max_steps,
        sessions: Mutex::new(sessions),
        cache_manager: Mutex::new(CacheManager::new(service.terminal().clone())),
    });

    // 将所有recv逻辑移到这个handle中
    let service_manager_for_recv = service_manager.clone();

    let _send_handle = tokio::task::spawn_blocking(move || {
        loop {
            println!("开始接收数据...");

            let Received { sessions, outputs } = service.recv();

            println!(
                "输出数量: {}, 完成会话数量: {}",
                outputs.len(),
                sessions.len()
            );

            // 先处理会话结束
            //TODO 变为并行版本
            if !sessions.is_empty() {
                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                let mut cache_manager_guard =
                    service_manager_for_recv.cache_manager.lock().unwrap();

                for (session, reason) in sessions {
                    let session_info = sessions_guard.remove(&session.id).unwrap();

                    let SessionInfo { tokens, .. } = session_info;
                    // 根据不同的结束原因进行处理
                    match reason {
                        ReturnReason::Finish => {
                            // 正常完成，插回cache
                            cache_manager_guard.insert(tokens, session.cache);
                        }
                        ReturnReason::Overflow => {
                            cache_manager_guard.insert(tokens, session.cache);
                            // TODO sender 信息需要发送给前端
                            todo!()
                        }
                        ReturnReason::NoDecode => {
                            cache_manager_guard.insert(tokens, session.cache);
                            // TODO sender 信息需要发送给前端
                            todo!()
                        }
                    }
                }
            }

            // 处理输出
            //TODO 变为并行版本
            for (session_id, tokens) in outputs {
                let now = std::time::Instant::now();
                println!(
                    "处理会话 {:?} 的输出，tokens长度: {} - 时间: {:?}",
                    session_id,
                    tokens.len(),
                    now
                );
                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                let session_info = sessions_guard.get_mut(&session_id).unwrap();
                // 更新session_info
                session_info.tokens.extend(&tokens);

                let text = service_manager_for_recv
                    .terminal
                    .decode(&tokens, &mut session_info.buf);
                info!("发送文本: {}", text);
                println!("解码文本: '{}' - 时间: {:?}", text, now);

                if session_info.sender.send(text.to_string()).is_err() {
                    println!("发送失败，接收端已关闭");
                    // 发送失败，可能是接收端已关闭
                    break;
                }
            }
        }
    });

    let app = App(service_manager);

    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        info!("ready to accept");
        let (stream, x) = listener.accept().await?;
        info!("listen from {x}");
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                warn!("Error serving connection: {err:?}")
            }
        });
    }
}

struct SessionInfo {
    sender: UnboundedSender<String>,
    buf: TextBuf,
    tokens: Vec<utok>,
    /// 剩余步数
    remaining_steps: usize,
}

struct ServiceManager {
    terminal: Terminal,
    max_steps: usize,
    sessions: Mutex<BTreeMap<SessionId, SessionInfo>>,
    cache_manager: Mutex<CacheManager>,
}

#[derive(Clone)]
struct App(Arc<ServiceManager>);

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let service_manager = self.0.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, V1_CHAT_COMPLETIONS) => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body);
                Ok(match req {
                    Ok(completions) => complete_chat(completions, service_manager),
                    Err(e) => error(Error::WrongJson(e)),
                })
            }),
            // Return 404 Not Found for other routes.
            (method, uri) => {
                let msg = Error::not_found(method, uri);
                Box::pin(async move { Ok(error(msg)) })
            }
        }
    }
}

fn complete_chat(
    completions: CreateChatCompletionRequest,
    service_manager: Arc<ServiceManager>,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    let CreateChatCompletionRequest {
        messages,
        max_tokens,
        ..
    } = completions;
    let (sender, receiver) = mpsc::unbounded_channel();

    let max_tokens = max_tokens.map_or(service_manager.max_steps, |max_tokens| max_tokens as usize);
    //TODO 需要从completions中获取
    let sample_args = Default::default();

    info!("completions: {messages:#?}");

    // 用于持有所有权
    let content_list = messages
        .iter()
        .map(|message| match message {
            ChatCompletionRequestMessage::User(user_message) => user_message.content.to_string(),
            ChatCompletionRequestMessage::System(system_message) => {
                system_message.content.to_string()
            }
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();

    let messages = messages
        .iter()
        .zip(&content_list)
        .map(|(message, content)| match message {
            ChatCompletionRequestMessage::User(_) => Message::user(content.as_str()),
            ChatCompletionRequestMessage::System(_) => Message::system(content.as_str()),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let text = service_manager.terminal.render(&messages);
    let tokens = service_manager.terminal.tokenize(&text);

    let (id, tokens) = service_manager
        .cache_manager
        .lock()
        .unwrap()
        .send(tokens, sample_args);

    assert!(
        service_manager
            .sessions
            .lock()
            .unwrap()
            .insert(
                id,
                SessionInfo {
                    sender,
                    tokens,
                    buf: TextBuf::new(),
                    remaining_steps: max_tokens,
                },
            )
            .is_none()
    );

    text_stream(UnboundedReceiverStream::new(receiver))
}

#[cfg(test)]
mod test {
    use super::{V1_CHAT_COMPLETIONS, start_infer_service};
    use log::{info, trace};
    use openai_struct::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, ModelIdsShared,
    };
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use tokio::time::Duration;
    use tokio_stream::StreamExt;

    const PORT: u16 = 24526;

    //TODO 目前是分开的形式
    #[test]
    fn test_post() {
        crate::logger::init();

        let Some(path) = std::env::var_os("TEST_MODEL") else {
            println!("TEST_MODE not set");
            return;
        };

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let _handle = tokio::spawn(start_infer_service(
                    path.into(),
                    PORT,
                    [0].into(),
                    256,
                    false,
                ));

                _handle.await.unwrap().unwrap();
            })
    }

    #[test]
    fn test_post_send() {
        crate::logger::init();

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let client = reqwest::Client::new();

                info!(
                    "Runtime workers = {}",
                    tokio::runtime::Handle::current().metrics().num_workers()
                );

                let mut headers: HeaderMap = HeaderMap::new();
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

                let req_body = serde_json::to_string(&CreateChatCompletionRequest {
                    model: ModelIdsShared {},
                    messages: vec![ChatCompletionRequestMessage::User(
                        openai_struct::ChatCompletionRequestUserMessage {
                            content: serde_json::Value::String("Tell a story".into()),
                            name: None,
                        },
                    )],
                    metadata: None,
                    service_tier: None,
                    audio: None,
                    function_call: None,
                    functions: None,
                    max_completion_tokens: None,
                    max_tokens: Some(256),
                    modalities: None,
                    n: None,
                    parallel_tool_calls: None,
                    prediction: None,
                    reasoning_effort: None,
                    response_format: None,
                    store: None,
                    tool_choice: None,
                    tools: None,
                    top_logprobs: None,
                    web_search_options: None,
                    frequency_penalty: None,
                    logit_bias: None,
                    logprobs: None,
                    presence_penalty: None,
                    seed: None,
                    stop: None,
                    stream: Some(true),
                    stream_options: None,
                    temperature: None,
                    top_p: None,
                    user: None,
                })
                .unwrap();

                let req = client
                    .post(format!("http://localhost:{PORT}{V1_CHAT_COMPLETIONS}"))
                    .headers(headers)
                    .body(req_body)
                    .timeout(Duration::from_secs(100));

                trace!("send req");
                let res = req.send().await.unwrap();
                info!("res: status={}, header={:#?}", res.status(), res.headers());

                if res.status().is_success() {
                    println!("开始读取流式响应...");

                    // 方法2: 使用流式读取 (正确的SSE处理方式)
                    let mut stream = res.bytes_stream();
                    let mut chunk_count = 0;
                    let mut total_text = String::new();

                    while let Some(item) = stream.next().await {
                        chunk_count += 1;
                        let now = std::time::Instant::now();
                        println!("收到第 {} 个数据块 - 时间: {:?}", chunk_count, now);
                        let bytes = item.unwrap();
                        let text = std::str::from_utf8(&bytes).unwrap();
                        println!("原始数据: {:?}", text);

                        // 解析SSE格式
                        for line in text.lines() {
                            if line.starts_with("data: ") {
                                let content = &line[6..];
                                println!("SSE数据: '{}'", content);
                                total_text.push_str(content);
                            } else if !line.is_empty() {
                                println!("非SSE行: {}", line);
                            }
                        }
                    }

                    println!("流式响应结束，共收到 {} 个数据块", chunk_count);
                    println!("完整文本: '{}'", total_text);
                } else {
                    println!("{res:?}");
                    let text = res.bytes().await.unwrap();
                    let text = std::str::from_utf8(&text).unwrap();
                    println!("body: {text}")
                }
            })
    }
}
