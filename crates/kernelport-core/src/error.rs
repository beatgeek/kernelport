use std::fmt;

/// A failure caused by the caller's request, not by the server or the model.
///
/// Backends attach this to an `anyhow::Error` so the serving layer can map the
/// failure to a client-facing status (`INVALID_ARGUMENT`) instead of collapsing
/// every error into `INTERNAL`. Anything not wrapped in this type is treated as
/// a server-side fault.
#[derive(Debug, Clone)]
pub struct InvalidRequest(pub String);

impl InvalidRequest {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// Build an `anyhow::Error` carrying this classification.
    pub fn err(message: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(Self::new(message))
    }
}

impl fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidRequest {}

/// True when `err` (or any error in its chain) was classified as caller-caused.
pub fn is_invalid_request(err: &anyhow::Error) -> bool {
    err.chain().any(|c| c.is::<InvalidRequest>())
}
