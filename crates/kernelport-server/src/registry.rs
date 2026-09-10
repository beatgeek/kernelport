use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use kernelport_backend_helion::HelionBackend;
use kernelport_backend_ort::OrtBackend;
use kernelport_core::{Backend, BackendModel, Device, IOName, ModelArtifact, Tensor};

pub struct LoadedModel {
    pub model: Mutex<Box<dyn BackendModelAdapter>>,
}

pub trait BackendModelAdapter: Send {
    fn infer(&mut self, inputs: Vec<(IOName, Tensor)>) -> Result<Vec<(IOName, Tensor)>>;
}

impl<T: BackendModel> BackendModelAdapter for T {
    fn infer(&mut self, inputs: Vec<(IOName, Tensor)>) -> Result<Vec<(IOName, Tensor)>> {
        BackendModel::infer_named(self, inputs)
    }
}

#[derive(Default)]
pub struct ModelRegistry {
    models: HashMap<String, Arc<LoadedModel>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn load_onnx(
        &mut self,
        name: &str,
        path: std::path::PathBuf,
        device: Device,
    ) -> Result<()> {
        let backend = OrtBackend::new();
        let artifact = ModelArtifact::OnnxPath(path);
        let model = backend.load(&artifact, device)?;

        let loaded = LoadedModel {
            model: Mutex::new(Box::new(model)),
        };

        self.models.insert(name.to_string(), Arc::new(loaded));
        Ok(())
    }

    pub fn load_helion(
        &mut self,
        name: &str,
        addr: String,
        model: String,
        device: Device,
    ) -> Result<()> {
        let backend = HelionBackend::new();
        let artifact = ModelArtifact::HelionGrpc { addr, model };
        let model = backend.load(&artifact, device)?;

        let loaded = LoadedModel {
            model: Mutex::new(Box::new(model)),
        };

        self.models.insert(name.to_string(), Arc::new(loaded));
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<LoadedModel>> {
        self.models.get(name).cloned()
    }
}
