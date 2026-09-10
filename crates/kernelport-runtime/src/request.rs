use kernelport_core::{is_invalid_request, IOName, Tensor};
use tokio::sync::oneshot;

/// Why an inference failed, carried far enough up the stack for the serving
/// layer to pick the right client-facing status code.
///
/// Keep the distinction: `InvalidArgument` means the caller sent something the
/// server correctly refused, `Internal` means the server or model faulted.
/// Collapsing the two makes bad requests indistinguishable from outages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferError {
    InvalidArgument(String),
    Internal(String),
}

impl InferError {
    /// Classify an `anyhow::Error` from a backend, preserving the full chain in
    /// the message so operators keep the context in logs.
    pub fn from_backend(err: &anyhow::Error) -> Self {
        let message = format!("{err:#}");
        if is_invalid_request(err) {
            Self::InvalidArgument(message)
        } else {
            Self::Internal(format!("model inference failed: {message}"))
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::InvalidArgument(m) | Self::Internal(m) => m,
        }
    }
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

#[derive(Debug)]
pub struct InferenceRequest {
    pub model: String,
    pub version: Option<String>,
    pub inputs: Vec<(IOName, Tensor)>,
    pub deadline: std::time::Instant,
    pub resp_tx: oneshot::Sender<Result<InferenceResponse, InferError>>,
}

#[derive(Debug)]
pub struct InferenceResponse {
    pub outputs: Vec<(IOName, Tensor)>,
    pub timings: Timings,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Timings {
    pub queued_us: u64,
    pub batched_us: u64,
    pub backend_us: u64,
}
