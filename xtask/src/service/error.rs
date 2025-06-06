use hyper::{Method, StatusCode};
use serde::Serialize;

#[derive(Debug)]
pub(crate) enum Error {
    WrongJson(serde_json::Error),
    NotFound(NotFoundError),
    MsgNotSupported(MsgNotSupportedError),
}

#[derive(Serialize, Debug)]
pub(crate) struct NotFoundError {
    method: String,
    uri: String,
}

#[derive(Serialize, Debug)]
pub(crate) struct MsgNotSupportedError {
    message: String,
}

impl Error {
    pub fn not_found(method: &Method, uri: &str) -> Self {
        Self::NotFound(NotFoundError {
            method: method.to_string(),
            uri: uri.into(),
        })
    }

    pub fn msg_not_supported(msg: &impl Serialize) -> Self {
        Self::MsgNotSupported(MsgNotSupportedError {
            message: serde_json::to_string_pretty(msg).unwrap(),
        })
    }

    #[inline]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::WrongJson(..) => StatusCode::BAD_REQUEST,
            Self::NotFound(..) => StatusCode::NOT_FOUND,
            Self::MsgNotSupported(..) => StatusCode::BAD_REQUEST,
        }
    }

    #[inline]
    pub fn body(&self) -> String {
        match self {
            Self::WrongJson(e) => e.to_string(),
            Self::NotFound(e) => serde_json::to_string(&e).unwrap(),
            Self::MsgNotSupported(e) => serde_json::to_string(&e).unwrap(),
        }
    }
}
