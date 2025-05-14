mod response;
mod schemas;

use http_body_util::{BodyExt, Empty, combinators::BoxBody};
use hyper::{
    Method, Request, Response, StatusCode,
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
};
use hyper_util::rt::TokioIo;
use llama_cu::{Handle, Session};
use log::{error, info, warn};
use response::{error, text_stream};
use schemas::Infer;
use std::sync::Mutex;
use std::{
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::PathBuf,
    pin::Pin,
    sync::Arc,
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Args)]
pub struct ServiceArgs {
    model: PathBuf,
    port: u16,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_steps: Option<usize>,
}

impl ServiceArgs {
    pub fn service(self) {
        let Self {
            model,
            port,
            gpus,
            max_steps,
        } = self;
        // 启动 tokio 运行时
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(start_infer_service(model, port, gpus, max_steps))
            .unwrap();
        runtime.shutdown_background();
    }
}

async fn start_infer_service(
    model: PathBuf,
    port: u16,
    gpus: Option<String>,
    max_steps: Option<usize>,
) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let gpus = gpus
        .map(|gpus| {
            gpus.split(',')
                .filter_map(|gpu| match gpu.parse::<i32>() {
                    Ok(gpu) => Some(gpu),
                    Err(e) => {
                        error!("{}", e);
                        None
                    }
                })
                .collect()
        })
        .unwrap_or(vec![0].into_boxed_slice());
    let max_steps = max_steps.unwrap_or(usize::MAX);

    let session_with_handle = Mutex::new(Session::new(model, gpus, max_steps));

    let app = App(Arc::new(ServiceManager {
        session_with_handle,
    }));
    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        let (stream, _) = listener.accept().await?;
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                warn!("Error serving connection: {err:?}");
            }
        });
    }
}

struct ServiceManager {
    session_with_handle: Mutex<(Session, Handle)>,
}

#[derive(Clone)]
struct App(Arc<ServiceManager>);

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let session_manager = self.0.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, "/infer") => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice::<Infer>(&whole_body);
                Ok(match req {
                    Ok(Infer { prompt }) => {
                        println!("{prompt}");

                        let session_manager = session_manager.clone();
                        let (sender, receiver) = mpsc::unbounded_channel();

                        let mut session_with_handle =
                            session_manager.session_with_handle.lock().unwrap();
                        let busy_session_receiver =
                            session_with_handle.0.send(prompt).into_receiver();
                        tokio::spawn(async move {
                            println!("start");
                            while let Ok(response) = busy_session_receiver.recv() {
                                println!("{response}");
                                sender.send(response).unwrap();
                            }
                            sender.closed().await;
                        });

                        text_stream(UnboundedReceiverStream::new(receiver))
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
