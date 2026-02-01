use kernelport_backend_helion::HelionBackend;
use kernelport_core::{Backend, BackendModel, Device, ModelArtifact};

#[tokio::test(flavor = "multi_thread")]
async fn loads_helion_spec() {
    let backend = HelionBackend::new();
    let artifact = ModelArtifact::HelionGrpc {
        addr: "http://127.0.0.1:50061".to_string(),
        model: "softmax_two_pass".to_string(),
    };

    let model = backend
        .load(&artifact, Device::Cuda { device_id: 0 })
        .expect("load helion model");

    assert_eq!(model.spec().inputs.len(), 1);
    assert_eq!(model.spec().outputs.len(), 1);
    assert_eq!(model.spec().inputs[0].name.0, "x");
    assert_eq!(model.spec().outputs[0].name.0, "y");
}
