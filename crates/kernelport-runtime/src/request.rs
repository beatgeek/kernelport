use kernelport_core::{IOName, Tensor};
use tokio::sync::oneshot;

#[derive(Debug)]
pub struct InferenceRequest {
    pub model: String,
    pub version: Option<String>,
    pub inputs: Vec<(IOName, Tensor)>,
    pub deadline: std::time::Instant,
    pub resp_tx: oneshot::Sender<InferenceResponse>,
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
