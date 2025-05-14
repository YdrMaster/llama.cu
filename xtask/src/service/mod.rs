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
use response::{error, text_stream};
use schemas::Infer;
use std::{
    ffi::c_int,
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::PathBuf,
    pin::Pin,
    sync::Arc,
};
use tokio::{
    net::TcpListener,
    sync::{Mutex, mpsc},
};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Args)]
pub struct ServiceArgs {
    #[clap(flatten)]
    base: BaseArgs,
    port: u16,
}

impl ServiceArgs {
    pub fn service(self) {
        let Self { base, port } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        // 启动 tokio 运行时
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(start_infer_service(base.model, port, gpus, max_steps))
            .unwrap();
        runtime.shutdown_background();
    }
}

async fn start_infer_service(
    model: PathBuf,
    port: u16,
    gpus: Box<[c_int]>,
    max_steps: usize,
) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let (session, _handle) = Session::new(model, gpus, max_steps);
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
            (&Method::POST, "/infer") => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice::<Infer>(&whole_body);
                Ok(match req {
                    Ok(Infer { prompt }) => {
                        let (sender, receiver) = mpsc::unbounded_channel();
                        tokio::spawn(async move {
                            let mut session = session.lock().await;
                            let busy_session = session.send(prompt, true);
                            while let Some(response) = busy_session.receive() {
                                sender.send(response).unwrap()
                            }
                            sender.closed().await
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
