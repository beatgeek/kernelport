use anyhow::Result;
use bytes::Bytes;
use kernelport_core::{
    Backend, BackendCapabilities, BackendModel, Device, DType, ModelArtifact, ModelSpec, Shape, Tensor,
};

pub struct OrtBackend;

impl OrtBackend {
    pub fn new() -> Self {
        Self
    }
}

pub struct OrtModel {
    spec: ModelSpec,
    _device: Device,
    _artifact: ModelArtifact,
}

impl Backend for OrtBackend {
    type Model = OrtModel;

    fn name(&self) -> &'static str {
        "onnxruntime"
    }

    fn load(&self, artifact: &ModelArtifact, device: Device) -> Result<Self::Model> {
        // v0: stub spec. Replace with actual model introspection later.
        let spec = ModelSpec {
            inputs: vec![],
            outputs: vec![],
            max_batch: 32,
        };
        Ok(OrtModel {
            spec,
            _device: device,
            _artifact: artifact.clone(),
        })
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_dynamic_shapes: true,
            prefers_nchw: false,
            allows_cuda_graphs: false,
        }
    }
}

impl BackendModel for OrtModel {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn infer(&mut self, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // v0: dummy output
        let out = Tensor::from_cpu_bytes(DType::U8, Shape::from_slice(&[1]), Bytes::from_static(&[0u8]));
        Ok(vec![out])
    }
}

