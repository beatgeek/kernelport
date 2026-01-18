use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use kernelport_backend_ort::OrtBackend;
use kernelport_core::{
    Backend, BackendModel, DType, Device, ModelArtifact, Shape, Tensor, TensorStorage,
};

#[test]
fn ort_identity_cpu() -> Result<()> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/identity.onnx");

    let backend = OrtBackend::new();
    let mut model = backend.load(&ModelArtifact::OnnxPath(model_path), Device::Cpu)?;
    let spec = model.spec();

    let input_spec = spec.inputs.first().context("missing model input spec")?;
    ensure!(
        input_spec.dtype == DType::F32,
        "expected f32 identity model"
    );

    let mut shape = input_spec
        .dims
        .iter()
        .map(|d| d.unwrap_or(3))
        .collect::<Vec<_>>();
    if shape.is_empty() {
        shape.push(3);
    }

    let numel = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let input = Tensor::from_cpu_bytes(
        DType::F32,
        Shape::from_slice(&shape),
        bytes_from_slice(&data),
    );

    let outputs = model.infer(vec![input])?;
    let out = outputs.first().context("missing model output")?;
    ensure!(out.desc.dtype == DType::F32, "expected f32 output");

    let out_bytes = tensor_bytes(out)?;
    let out_vals = bytes_to_f32(out_bytes)?;
    assert_eq!(out_vals, data);

    Ok(())
}

fn tensor_bytes(tensor: &Tensor) -> Result<&Bytes> {
    match &tensor.storage {
        TensorStorage::CpuBytes(bytes) => Ok(bytes),
        TensorStorage::CpuPinned(p) => Ok(&p.bytes),
        TensorStorage::CudaDevice(_) => anyhow::bail!("CUDA output not supported in CPU test"),
    }
}

fn bytes_to_f32(bytes: &Bytes) -> Result<Vec<f32>> {
    ensure!(
        bytes.len().is_multiple_of(4),
        "f32 output has invalid byte length"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn bytes_from_slice<T>(slice: &[T]) -> Bytes {
    let byte_len = std::mem::size_of_val(slice);
    let ptr = slice.as_ptr().cast::<u8>();
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    Bytes::copy_from_slice(bytes)
}
