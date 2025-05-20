use hyper::StatusCode;

#[derive(Debug)]
pub(crate) enum Error {
    WrongJson(serde_json::Error),
}

#[derive(serde::Serialize)]
struct ErrorBody {
    status: u16,
    code: u16,
    message: String,
}

impl Error {
    #[inline]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::WrongJson(_) => StatusCode::BAD_REQUEST,
        }
    }

    #[inline]
    pub fn body(&self) -> serde_json::Value {
        macro_rules! error {
            ($code:expr, $msg:expr) => {
                ErrorBody {
                    status: self.status().as_u16(),
                    code: $code,
                    message: $msg.into(),
                }
            };
        }

        #[inline]
        fn json(v: impl serde::Serialize) -> serde_json::Value {
            serde_json::to_value(v).unwrap()
        }

        match self {
            Self::WrongJson(e) => json(error!(0, e.to_string())),
        }
    }
}
