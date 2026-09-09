use anyhow::Result;

use crate::{Device, IOName, ModelArtifact, ModelSpec, Tensor};

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

    /// Preserve the wire contract; positional backends use their declared IO order.
    fn infer_named(&mut self, inputs: Vec<(IOName, Tensor)>) -> Result<Vec<(IOName, Tensor)>> {
        let mut inputs = inputs;
        let mut ordered = Vec::new();
        for spec in &self.spec().inputs {
            let index = inputs
                .iter()
                .position(|(name, _)| name == &spec.name)
                .ok_or_else(|| anyhow::anyhow!("missing input: {}", spec.name.0))?;
            ordered.push(inputs.remove(index).1);
        }
        anyhow::ensure!(inputs.is_empty(), "unexpected or duplicate input names");
        let names: Vec<_> = self.spec().outputs.iter().map(|s| s.name.clone()).collect();
        let outputs = self.infer(ordered)?;
        anyhow::ensure!(
            names.len() == outputs.len(),
            "output count does not match model spec"
        );
        Ok(names.into_iter().zip(outputs).collect())
    }
}
