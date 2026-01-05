#[derive(Clone, Debug)]
pub enum ModelArtifact {
    OnnxPath(std::path::PathBuf),
    TensorRtEnginePath(std::path::PathBuf),
    TorchScriptPath(std::path::PathBuf),
    TfSavedModelDir(std::path::PathBuf),
}

