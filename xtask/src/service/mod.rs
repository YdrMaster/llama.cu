mod error;
mod openai;
mod response;

use crate::BaseArgs;
use error::Error;
use http_body_util::{BodyExt, Empty, combinators::BoxBody};
use hyper::{
    Method, Request, Response, StatusCode,
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
};
use hyper_util::rt::TokioIo;
use llama_cu::Session;
use log::{info, warn};
use openai::{Completions, CompletionsChoice, CompletionsResponse, V1_COMPLETIONS_OBJECT};
use response::{error, text_stream};
use std::{
    ffi::c_int,
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex, atomic::AtomicUsize},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;

use std::sync::atomic::Ordering::SeqCst;
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

    let (session, _handle) = Session::new(model, gpus, max_steps, use_cuda_graph);
    let app = App(Arc::new(Mutex::new(session)));

    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
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

#[derive(Clone)]
struct App(Arc<Mutex<Session>>);

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let session = self.0.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, openai::V1_COMPLETIONS) => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body);
                Ok(match req {
                    Ok(Completions { prompt, model }) => {
                        let (sender, receiver) = mpsc::unbounded_channel();
                        tokio::task::spawn_blocking(move || {
                            let mut session = session.lock().unwrap();
                            let busy_session = session.send(prompt, true);

                            static ID: AtomicUsize = AtomicUsize::new(0);

                            let id = format!("InfiniLM-{:#x}", ID.fetch_add(1, SeqCst));
                            let created = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs() as usize;

                            while let Some(response) = busy_session.receive() {
                                let response = CompletionsResponse {
                                    id: id.clone(),
                                    choices: vec![CompletionsChoice {
                                        index: 0,
                                        text: response,
                                    }],
                                    created,
                                    model: model.clone(),
                                    object: V1_COMPLETIONS_OBJECT.into(),
                                };
                                let msg = serde_json::to_string(&response).unwrap();
                                if sender.send(msg).is_err() {
                                    break;
                                }
                            }
                        });
                        text_stream(UnboundedReceiverStream::new(receiver))
                    }
                    Err(e) => error(Error::WrongJson(e)),
                })
            }),
            // Return 404 Not Found for other routes.
            _ => Box::pin(async move {
                Ok(Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(
                        Empty::<Bytes>::new()
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                    .unwrap())
            }),
        }
    }
}

#[test]
fn test_post() {
    use crate::macros::print_now;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use tokio_stream::StreamExt;

    let Some(path) = std::env::var_os("TEST_MODEL") else {
        println!("TEST_MODE not set");
        return;
    };
    const PORT: u16 = 27000;

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let client = reqwest::Client::new();

            let _handle = tokio::spawn(start_infer_service(
                path.into(),
                PORT,
                [0].into(),
                256,
                false,
            ));

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            let req = client
                .post(format!("http://localhost:{PORT}{}", openai::V1_COMPLETIONS))
                .headers(headers)
                .body(
                    serde_json::to_string(&Completions {
                        prompt: "Once upon a time,".into(),
                        model: "model".into(),
                    })
                    .unwrap(),
                );

            let res = req.send().await.unwrap();

            if res.status().is_success() {
                let mut stream = res.bytes_stream();
                while let Some(item) = stream.next().await {
                    let text = item.unwrap();
                    let text = std::str::from_utf8(&text).unwrap();
                    print_now!("{text}")
                }
            } else {
                println!("{res:?}");
                let text = res.bytes().await.unwrap();
                let text = std::str::from_utf8(&text).unwrap();
                println!("body: {text}")
            }
        })
}
