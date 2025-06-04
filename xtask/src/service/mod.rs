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
use llama_cu::{
    Message, Received, ReturnReason, SampleArgs, Service, SessionId, Terminal, TextBuf, utok,
};
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
    time::Duration,
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
            let Received { sessions, outputs } = service.recv(Duration::MAX);

            // 先处理输出
            for (session_id, tokens) in outputs {
                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                let session_info = sessions_guard.get_mut(&session_id).unwrap();
                // 更新session_info
                session_info.tokens.extend(&tokens);

                let text = service_manager_for_recv
                    .terminal
                    .decode(&tokens, &mut session_info.buf);
                info!("发送文本: {text:?}");

                if session_info.sender.send(text).is_err() {
                    todo!("客户端关闭")
                }
            }

            // 处理会话结束
            if !sessions.is_empty() {
                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                let mut cache_manager_guard =
                    service_manager_for_recv.cache_manager.lock().unwrap();

                for (session, reason) in sessions {
                    let SessionInfo { tokens, .. } = sessions_guard.remove(&session.id).unwrap();
                    // 根据不同的结束原因进行处理
                    match reason {
                        // 正常完成，插回cache
                        ReturnReason::Finish => cache_manager_guard.insert(tokens, session.cache),
                        ReturnReason::Overflow => todo!(),
                    }
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
        temperature,
        top_p,
        ..
    } = completions;
    let (sender, receiver) = mpsc::unbounded_channel();

    let max_steps = max_tokens.map_or(service_manager.max_steps, |n| n as usize);
    let sample_args =
        SampleArgs::new(temperature.unwrap_or(0.), top_p.unwrap_or(1.), usize::MAX).unwrap();

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

    let (id, tokens) =
        service_manager
            .cache_manager
            .lock()
            .unwrap()
            .send(tokens, sample_args, max_steps);

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
                },
            )
            .is_none()
    );

    text_stream(UnboundedReceiverStream::new(receiver))
}

#[cfg(test)]
mod test {
    use super::*;
    use log::{info, trace, warn};
    use openai_struct::ModelIdsShared;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::time::Instant;
    use tokio::time::Duration;
    use tokio_stream::StreamExt;

    const PORT: u16 = 24526;
    const CONCURRENT_REQUESTS: usize = 10;

    fn requset_body_chat(prompt: &str) -> String {
        serde_json::to_string(&CreateChatCompletionRequest {
            model: ModelIdsShared {},
            messages: vec![ChatCompletionRequestMessage::User(
                openai_struct::ChatCompletionRequestUserMessage {
                    content: serde_json::Value::String(prompt.into()),
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
        .unwrap()
    }

    fn create_client_with_headers() -> (reqwest::Client, HeaderMap) {
        let client = reqwest::Client::new();
        let mut headers: HeaderMap = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        (client, headers)
    }

    async fn send_single_request(
        client: &reqwest::Client,
        headers: &HeaderMap,
        req_body: String,
        index: Option<usize>,
    ) -> Result<(usize, usize, usize, Duration), String> {
        let task_start = Instant::now();
        let index = index.unwrap_or(0);

        if index > 0 {
            trace!("任务 {} 开始", index);
        }

        let req = client
            .post(format!("http://localhost:{PORT}{V1_CHAT_COMPLETIONS}"))
            .headers(headers.clone())
            .body(req_body)
            .timeout(Duration::from_secs(100));

        match req.send().await {
            Ok(res) => {
                let status = res.status();
                if index > 0 {
                    info!(
                        "任务 {} - 响应状态: {}, 耗时: {:?}",
                        index,
                        status,
                        task_start.elapsed()
                    );
                } else {
                    info!("响应状态: {}, header={:#?}", status, res.headers());
                }

                if status.is_success() {
                    if index > 0 {
                        trace!("任务 {} 开始读取流式响应...", index);
                    } else {
                        trace!("开始读取流式响应...");
                    }

                    let mut stream = res.bytes_stream();
                    let mut chunk_count = 0;
                    let mut total_text = String::new();

                    while let Some(item) = stream.next().await {
                        chunk_count += 1;
                        match item {
                            Ok(bytes) => {
                                let text = std::str::from_utf8(&bytes).unwrap_or("<invalid utf8>");

                                if index > 0 {
                                    trace!(
                                        "任务 {} 收到第 {} 个数据块: {:?}",
                                        index, chunk_count, text
                                    );
                                } else {
                                    let now = Instant::now();
                                    trace!("收到第 {} 个数据块 - 时间: {:?}", chunk_count, now);
                                    trace!("原始数据: {:?}", text);
                                }

                                // 解析 SSE 格式
                                for line in text.lines() {
                                    if let Some(line) = line.strip_prefix("data: ") {
                                        total_text.push_str(line);
                                        if index == 0 {
                                            trace!("SSE 行: `{}`", line);
                                        }
                                    } else if !line.is_empty() {
                                        total_text.push_str(line);
                                        if index == 0 {
                                            trace!("非 SSE 行: `{}`", line);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("任务 {} 读取流时出错: {:?}", index, e);
                                break;
                            }
                        }
                    }

                    if index > 0 {
                        info!(
                            "任务 {} 完成 - 总耗时: {:?}, 数据块数: {}, 文本长度: {}",
                            index,
                            task_start.elapsed(),
                            chunk_count,
                            total_text.len()
                        );
                    } else {
                        println!("流式响应结束，共收到 {} 个数据块", chunk_count);
                        println!("完整文本: {}", total_text);
                    }

                    Ok((index, chunk_count, total_text.len(), task_start.elapsed()))
                } else {
                    let error_text = res.text().await.unwrap_or_default();
                    if index > 0 {
                        warn!(
                            "任务 {} 失败 - 状态: {}, 错误: {}",
                            index, status, error_text
                        );
                    } else {
                        println!("body: {}", error_text);
                    }
                    Err(format!("HTTP错误: {}", status))
                }
            }
            Err(e) => {
                if index > 0 {
                    warn!("任务 {} 请求失败: {:?}", index, e);
                }
                Err(format!("请求错误: {:?}", e))
            }
        }
    }

    #[test]
    fn test_post_send() {
        crate::logger::init();

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let (client, headers) = create_client_with_headers();

                info!(
                    "Runtime workers = {}",
                    tokio::runtime::Handle::current().metrics().num_workers()
                );

                let req_body = requset_body_chat("Tell a story");

                trace!("send req");
                let _ = send_single_request(&client, &headers, req_body, None).await;
            })
    }

    #[test]
    fn test_post_send_multi() {
        crate::logger::init();

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let (client, headers) = create_client_with_headers();

                info!(
                    "Runtime workers = {}",
                    tokio::runtime::Handle::current().metrics().num_workers()
                );

                // 创建多个不同的请求内容
                let request_bodies = (0..CONCURRENT_REQUESTS)
                    .map(|i| requset_body_chat(&format!("Tell me a story number {}", i + 1)))
                    .collect::<Vec<_>>();

                let start_time = Instant::now();
                info!("开始发送 {} 个并发请求", CONCURRENT_REQUESTS);

                // 创建并发任务
                let tasks = request_bodies
                    .into_iter()
                    .enumerate()
                    .map(|(index, req_body)| {
                        let client = client.clone();
                        let headers = headers.clone();

                        tokio::spawn(async move {
                            send_single_request(&client, &headers, req_body, Some(index + 1)).await
                        })
                    })
                    .collect::<Vec<_>>();

                // 等待所有任务完成并统计结果
                let total_elapsed = start_time.elapsed();
                let mut successful_count = 0;
                let mut failed_count = 0;
                let mut total_chunks = 0;
                let mut total_text_length = 0;
                let mut max_duration = Duration::ZERO;
                let mut min_duration = Duration::MAX;

                for task in tasks {
                    match task.await {
                        Ok(Ok((index, chunks, text_len, duration))) => {
                            successful_count += 1;
                            total_chunks += chunks;
                            total_text_length += text_len;
                            max_duration = max_duration.max(duration);
                            min_duration = min_duration.min(duration);
                            trace!("任务 {} 成功完成", index);
                        }
                        Ok(Err(e)) => {
                            failed_count += 1;
                            warn!("任务失败: {}", e);
                        }
                        Err(e) => {
                            failed_count += 1;
                            warn!("任务执行出错: {:?}", e);
                        }
                    }
                }

                // 输出统计信息
                println!("\n=== 并发测试统计 ===");
                println!("总请求数: {}", CONCURRENT_REQUESTS);
                println!("成功请求数: {}", successful_count);
                println!("失败请求数: {}", failed_count);
                println!("总耗时: {:?}", total_elapsed);
                println!("最快请求: {:?}", min_duration);
                println!("最慢请求: {:?}", max_duration);
                println!(
                    "平均每请求耗时: {:?}",
                    total_elapsed / CONCURRENT_REQUESTS as u32
                );
                println!("总数据块数: {}", total_chunks);
                println!("总文本长度: {}", total_text_length);
                println!(
                    "成功率: {:.1}%",
                    (successful_count as f64 / CONCURRENT_REQUESTS as f64) * 100.0
                );

                if successful_count > 0 {
                    println!(
                        "平均每请求数据块数: {:.1}",
                        total_chunks as f64 / successful_count as f64
                    );
                    println!(
                        "平均每请求文本长度: {:.1}",
                        total_text_length as f64 / successful_count as f64
                    );
                }

                // 验证至少有一些请求成功
                assert!(successful_count > 0, "至少应该有一个请求成功");
            })
    }
}
