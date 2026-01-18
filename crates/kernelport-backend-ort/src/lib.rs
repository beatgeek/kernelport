use anyhow::{bail, ensure, Context, Result};
use bytes::Bytes;
use kernelport_core::{
    Backend, BackendCapabilities, BackendModel, DType, Device, IOName, ModelArtifact, ModelSpec,
    Shape, Tensor, TensorSpec, TensorStorage,
};
use ort::{
    session::{builder::SessionBuilder, Session, SessionInputValue},
    tensor::TensorElementType,
    value::{DynValue, ValueType},
};

pub struct OrtBackend;

impl OrtBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OrtBackend {
    fn default() -> Self {
        Self::new()
    }
}

pub struct OrtModel {
    spec: ModelSpec,
    session: Session,
    input_names: Vec<String>,
}

impl Backend for OrtBackend {
    type Model = OrtModel;

    fn name(&self) -> &'static str {
        "onnxruntime"
    }

    fn load(&self, artifact: &ModelArtifact, device: Device) -> Result<Self::Model> {
        let ModelArtifact::OnnxPath(path) = artifact else {
            bail!("onnxruntime backend expects an ONNX file path");
        };

        let builder = Session::builder()
            .context("failed to create ORT session builder")?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .context("failed to configure ORT session builder")?;

        let builder = configure_session_builder(builder, &device)?;

        let session = builder
            .commit_from_file(path)
            .context("failed to load ONNX model")?;

        let input_names = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let spec = build_model_spec(&session)?;

        Ok(OrtModel {
            spec,
            session,
            input_names,
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

    fn infer(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        ensure!(
            inputs.len() == self.input_names.len(),
            "expected {} inputs, got {}",
            self.input_names.len(),
            inputs.len()
        );

        let mut ort_inputs = Vec::with_capacity(inputs.len());
        for (name, input) in self.input_names.iter().zip(inputs) {
            let value = tensor_to_ort_value(input)?;
            ort_inputs.push((name.clone(), SessionInputValue::from(value)));
        }

        let outputs = self.session.run(ort_inputs)?;
        let mut out_tensors = Vec::with_capacity(outputs.len());
        for (_, value) in outputs.iter() {
            out_tensors.push(ort_value_to_tensor(&value)?);
        }

        Ok(out_tensors)
    }
}

fn build_model_spec(session: &Session) -> Result<ModelSpec> {
    let inputs = session
        .inputs
        .iter()
        .map(|input| tensor_spec_from_value_type(&input.name, &input.input_type))
        .collect::<Result<Vec<_>>>()?;

    let outputs = session
        .outputs
        .iter()
        .map(|output| tensor_spec_from_value_type(&output.name, &output.output_type))
        .collect::<Result<Vec<_>>>()?;

    Ok(ModelSpec {
        inputs,
        outputs,
        max_batch: 32,
    })
}

fn configure_session_builder(builder: SessionBuilder, device: &Device) -> Result<SessionBuilder> {
    match device {
        Device::Cpu => Ok(builder),
        Device::Cuda { device_id } => configure_cuda(builder, *device_id),
    }
}

fn configure_cuda(builder: SessionBuilder, device_id: u32) -> Result<SessionBuilder> {
    #[cfg(feature = "cuda")]
    {
        use ort::execution_providers::cuda::CUDAExecutionProvider;
        let ep = CUDAExecutionProvider::default()
            .with_device_id(device_id as i32)
            .build();
        builder
            .with_execution_providers([ep])
            .context("failed to enable ORT CUDA execution provider")
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (builder, device_id);
        bail!("CUDA requested but kernelport-backend-ort was built without the `cuda` feature")
    }
}

fn tensor_spec_from_value_type(name: &str, value_type: &ValueType) -> Result<TensorSpec> {
    let ValueType::Tensor { ty, shape, .. } = value_type else {
        bail!("unsupported non-tensor IO value type");
    };

    let dtype = ort_tensor_element_to_dtype(*ty)?;
    let dims = shape
        .iter()
        .map(|d| if *d < 0 { None } else { Some(*d as usize) })
        .collect::<Vec<_>>();

    Ok(TensorSpec {
        name: IOName(name.to_string()),
        dtype,
        rank: shape.len(),
        dims,
    })
}

fn ort_tensor_element_to_dtype(ty: TensorElementType) -> Result<DType> {
    match ty {
        TensorElementType::Float32 => Ok(DType::F32),
        TensorElementType::Float16 => Ok(DType::F16),
        TensorElementType::Int64 => Ok(DType::I64),
        TensorElementType::Int32 => Ok(DType::I32),
        TensorElementType::Uint8 => Ok(DType::U8),
        _ => bail!("unsupported tensor element type: {ty}"),
    }
}

fn dtype_byte_size(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F32 => Ok(4),
        DType::F16 => Ok(2),
        DType::I64 => Ok(8),
        DType::I32 => Ok(4),
        DType::U8 => Ok(1),
    }
}

fn tensor_bytes(tensor: &Tensor) -> Result<&Bytes> {
    match &tensor.storage {
        TensorStorage::CpuBytes(bytes) => Ok(bytes),
        TensorStorage::CpuPinned(p) => Ok(&p.bytes),
        TensorStorage::CudaDevice(_) => bail!("CUDA tensors are not supported in CPU backend"),
    }
}

fn tensor_to_ort_value(tensor: Tensor) -> Result<DynValue> {
    let bytes = tensor_bytes(&tensor)?;
    let shape: Vec<usize> = tensor.desc.shape.0.iter().copied().collect();
    let expected_bytes = tensor.desc.shape.numel() * dtype_byte_size(tensor.desc.dtype)?;
    ensure!(
        bytes.len() == expected_bytes,
        "input byte size mismatch: got {}, expected {}",
        bytes.len(),
        expected_bytes
    );

    let value = match tensor.desc.dtype {
        DType::F32 => {
            let data = bytes_to_f32(bytes)?;
            ort::value::Tensor::from_array((shape, data))?.into_dyn()
        }
        DType::I64 => {
            let data = bytes_to_i64(bytes)?;
            ort::value::Tensor::from_array((shape, data))?.into_dyn()
        }
        DType::I32 => {
            let data = bytes_to_i32(bytes)?;
            ort::value::Tensor::from_array((shape, data))?.into_dyn()
        }
        DType::U8 => {
            let data = bytes.to_vec();
            ort::value::Tensor::from_array((shape, data))?.into_dyn()
        }
        DType::F16 => bail!("f16 inputs are not supported yet"),
    };

    Ok(value)
}

fn ort_value_to_tensor(value: &ort::value::ValueRef<'_>) -> Result<Tensor> {
    let ValueType::Tensor { ty, shape, .. } = value.dtype() else {
        bail!("non-tensor outputs are not supported");
    };

    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    let kernel_shape = Shape::from_slice(&dims);

    match *ty {
        TensorElementType::Float32 => {
            let array = value.try_extract_array::<f32>()?;
            let slice = array.as_slice().context("non-contiguous output tensor")?;
            Ok(Tensor::from_cpu_bytes(
                DType::F32,
                kernel_shape,
                bytes_from_slice(slice),
            ))
        }
        TensorElementType::Int64 => {
            let array = value.try_extract_array::<i64>()?;
            let slice = array.as_slice().context("non-contiguous output tensor")?;
            Ok(Tensor::from_cpu_bytes(
                DType::I64,
                kernel_shape,
                bytes_from_slice(slice),
            ))
        }
        TensorElementType::Int32 => {
            let array = value.try_extract_array::<i32>()?;
            let slice = array.as_slice().context("non-contiguous output tensor")?;
            Ok(Tensor::from_cpu_bytes(
                DType::I32,
                kernel_shape,
                bytes_from_slice(slice),
            ))
        }
        TensorElementType::Uint8 => {
            let array = value.try_extract_array::<u8>()?;
            let slice = array.as_slice().context("non-contiguous output tensor")?;
            Ok(Tensor::from_cpu_bytes(
                DType::U8,
                kernel_shape,
                Bytes::copy_from_slice(slice),
            ))
        }
        TensorElementType::Float16 => bail!("f16 outputs are not supported yet"),
        _ => bail!("unsupported output tensor element type: {ty}"),
    }
}

#[allow(clippy::manual_is_multiple_of)]
fn bytes_to_f32(bytes: &Bytes) -> Result<Vec<f32>> {
    ensure!(bytes.len() % 4 == 0, "f32 input has invalid byte length");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[allow(clippy::manual_is_multiple_of)]
fn bytes_to_i64(bytes: &Bytes) -> Result<Vec<i64>> {
    ensure!(bytes.len() % 8 == 0, "i64 input has invalid byte length");
    Ok(bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect())
}

#[allow(clippy::manual_is_multiple_of)]
fn bytes_to_i32(bytes: &Bytes) -> Result<Vec<i32>> {
    ensure!(bytes.len() % 4 == 0, "i32 input has invalid byte length");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn bytes_from_slice<T>(slice: &[T]) -> Bytes {
    let byte_len = std::mem::size_of_val(slice);
    let ptr = slice.as_ptr().cast::<u8>();
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    Bytes::copy_from_slice(bytes)
}
