use anyhow::Result;

use crate::{Device, ModelArtifact, ModelSpec, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct BackendCapabilities {
    pub supports_dynamic_shapes: bool,
    pub prefers_nchw: bool,
    pub allows_cuda_graphs: bool,
}

pub trait Backend: Send + Sync + 'static {
    type Model: BackendModel;

    fn name(&self) -> &'static str;
    fn load(&self, artifact: &ModelArtifact, device: Device) -> Result<Self::Model>;
    fn capabilities(&self) -> BackendCapabilities;
}

pub trait BackendModel: Send + 'static {
    fn spec(&self) -> &ModelSpec;

    /// Inputs are already batched and (eventually) on the right device.
    fn infer(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;
}

