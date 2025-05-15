mod response;
mod schemas;

use crate::BaseArgs;
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
use regex::Regex;
use response::{error, text_stream};
use schemas::Infer;
use std::{
    collections::{HashMap, HashSet},
    ffi::c_int,
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;

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
        let max_steps = base.max_steps();
        let special = Regex::new(r"^(\d+)x(\d+)$").unwrap();
        if let Some(captures) = base.gpus.as_deref().and_then(|s| special.captures(s)) {
            let (_, [a, b]) = captures.extract();
            let n_service = a.parse::<usize>().unwrap();
            let dev_service = b.parse::<usize>().unwrap();
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(start_infer_service(
                    base.model,
                    port,
                    (0..n_service)
                        .map(|i| {
                            (i * dev_service..(i + 1) * dev_service)
                                .map(|i| i as c_int)
                                .collect()
                        })
                        .collect(),
                    max_steps,
                ))
                .unwrap()
        } else {
            let gpus = base.gpus();
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(start_infer_service(
                    base.model,
                    port,
                    [gpus].into(),
                    max_steps,
                ))
                .unwrap()
        }
    }
}

async fn start_infer_service(
    model: PathBuf,
    port: u16,
    gpuss: Box<[Box<[c_int]>]>,
    max_steps: usize,
) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let mut sessions = Vec::new();
    let mut handles = Vec::new();
    for gpus in gpuss {
        let (session, handle) = Session::new(model.clone(), gpus, max_steps);
        sessions.push(session);
        handles.push(handle);
    }

    let app = App(Arc::new(Manager::new(sessions)));

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
struct App(Arc<Manager>);

struct Manager {
    sessions: Box<[Arc<Mutex<Session>>]>,
    map: Mutex<HashMap<String, usize>>,
}

impl Manager {
    pub fn new(sessions: impl IntoIterator<Item = Session>) -> Self {
        Self {
            sessions: sessions.into_iter().map(Mutex::new).map(Arc::new).collect(),
            map: Default::default(),
        }
    }
}

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let manager = self.0.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, "/infer") => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice::<Infer>(&whole_body);
                Ok(match req {
                    Ok(Infer { prompt, id }) => {
                        if let [session] = &*manager.sessions {
                            start(session.clone(), prompt)
                        } else if let Some(id) = id {
                            use std::collections::hash_map::Entry::{Occupied, Vacant};
                            let mut map = manager.map.lock().unwrap();
                            let used = map.values().cloned().collect::<HashSet<_>>();
                            match map.entry(id) {
                                Occupied(entry) => {
                                    start(manager.sessions[entry.get().clone()].clone(), prompt)
                                }
                                Vacant(entry) => {
                                    let (i, session) = manager
                                        .sessions
                                        .iter()
                                        .enumerate()
                                        .find(|(i, _)| !used.contains(i))
                                        .expect("all sessions occupied");
                                    entry.insert(i);
                                    start(session.clone(), prompt)
                                }
                            }
                        } else {
                            error(schemas::Error::NeedId)
                        }
                    }
                    Err(e) => error(schemas::Error::WrongJson(e)),
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

fn start(session: Arc<Mutex<Session>>, prompt: String) -> Response<BoxBody<Bytes, hyper::Error>> {
    let (sender, receiver) = mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        let mut session = session.lock().unwrap();
        let busy_session = session.send(prompt, true);
        while let Some(response) = busy_session.receive() {
            if sender.send(response).is_err() {
                break;
            }
        }
    });
    text_stream(UnboundedReceiverStream::new(receiver))
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
                [[0].into()].into(),
                256,
            ));

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            let req = client
                .post(format!("http://localhost:{PORT}/infer"))
                .headers(headers)
                .body(
                    serde_json::to_string(&Infer {
                        prompt: "Once upon a time,".into(),
                        id: None,
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
