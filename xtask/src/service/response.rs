//! All HttpResponses in this App.

use super::schemas;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::{
    Response, StatusCode,
    body::{Bytes, Frame},
    header::CONTENT_TYPE,
};
use tokio_stream::{Stream, StreamExt};

pub fn text_stream(
    s: impl Stream<Item = String> + Send + Sync + 'static,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(StreamBody::new(s.map(|s| Ok(Frame::data(s.into())))).boxed())
        .unwrap()
}

pub fn error(e: schemas::Error) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(e.status())
        .header(CONTENT_TYPE, "application/json")
        .body(full(serde_json::to_string(&e.body()).unwrap()))
        .unwrap()
}

#[inline]
fn full(chunk: impl Into<Bytes>) -> BoxBody<Bytes, hyper::Error> {
    Full::new(chunk.into())
        .map_err(|never| match never {})
        .boxed()
}
