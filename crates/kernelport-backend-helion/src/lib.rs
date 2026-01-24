use std::time::Duration;

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use kernelport_core::{
    Backend, BackendCapabilities, BackendModel, DType, Device, IOName, ModelArtifact, ModelSpec,
    Shape, Tensor, TensorSpec, TensorStorage,
};
use kernelport_proto::kernelport::v1 as pb;
use kernelport_proto::kernelport::v1::inference_service_client::InferenceServiceClient;
use tonic::transport::{Channel, Endpoint};

pub struct HelionBackend;

impl HelionBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HelionBackend {
    fn default() -> Self {
        Self::new()
    }
}

pub struct HelionModel {
    spec: ModelSpec,
    client: InferenceServiceClient<Channel>,
    model: String,
    timeout: Duration,
}

impl Backend for HelionBackend {
    type Model = HelionModel;

    fn name(&self) -> &'static str {
        "helion"
    }

    fn load(&self, artifact: &ModelArtifact, _device: Device) -> Result<Self::Model> {
        let ModelArtifact::HelionGrpc { addr, model } = artifact else {
            bail!("helion backend expects a HelionGrpc artifact");
        };

        let endpoint = Endpoint::from_shared(addr.clone())
            .context("invalid helion gRPC address")?
            .connect_lazy();
        let client = InferenceServiceClient::new(endpoint);

        // v0: assume softmax-like 2D f16 tensors with x->y names and dynamic dims.
        let spec = ModelSpec {
            inputs: vec![TensorSpec {
                name: IOName("x".to_string()),
                dtype: DType::F16,
                rank: 2,
                dims: vec![None, None],
            }],
            outputs: vec![TensorSpec {
                name: IOName("y".to_string()),
                dtype: DType::F16,
                rank: 2,
                dims: vec![None, None],
            }],
            max_batch: 1,
        };

        Ok(HelionModel {
            spec,
            client,
            model: model.clone(),
            timeout: Duration::from_secs(120),
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

impl BackendModel for HelionModel {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn infer(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let mut pb_inputs = Vec::with_capacity(inputs.len());
        for (idx, input) in inputs.into_iter().enumerate() {
            pb_inputs.push(tensor_to_pb(&format!("input{idx}"), input)?);
        }

        let mut request = tonic::Request::new(pb::InferRequest {
            model: self.model.clone(),
            inputs: pb_inputs,
        });
        request.set_timeout(self.timeout);

        let mut client = self.client.clone();
        let response: tonic::Response<pb::InferResponse> = tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(async { client.infer(request).await })
        })
        .context("helion inference request failed")?;
        let response = response.into_inner();

        let mut outputs = Vec::with_capacity(response.outputs.len());
        for output in response.outputs {
            outputs.push(pb_to_tensor(output)?);
        }

        Ok(outputs)
    }
}

fn tensor_to_pb(name: &str, tensor: Tensor) -> Result<pb::Tensor> {
    let data = match tensor.storage {
        TensorStorage::CpuBytes(bytes) => bytes,
        TensorStorage::CpuPinned(p) => p.bytes,
        TensorStorage::CudaDevice(_) => bail!("helion backend only supports CPU tensors"),
    };

    Ok(pb::Tensor {
        name: name.to_string(),
        dtype: to_proto_dtype(tensor.desc.dtype) as i32,
        shape: tensor.desc.shape.0.iter().map(|d| *d as i64).collect(),
        data: data.to_vec(),
    })
}

fn pb_to_tensor(tensor: pb::Tensor) -> Result<Tensor> {
    let dtype = parse_dtype(tensor.dtype)?;
    let shape: Vec<usize> = tensor
        .shape
        .into_iter()
        .map(|d| usize::try_from(d).unwrap_or(0))
        .collect();

    Ok(Tensor::from_cpu_bytes(
        dtype,
        Shape::from_slice(&shape),
        Bytes::from(tensor.data),
    ))
}

fn parse_dtype(raw: i32) -> Result<DType> {
    let dtype = pb::DType::try_from(raw).context("unknown dtype enum value")?;
    Ok(match dtype {
        pb::DType::F32 => DType::F32,
        pb::DType::F16 => DType::F16,
        pb::DType::I64 => DType::I64,
        pb::DType::I32 => DType::I32,
        pb::DType::U8 => DType::U8,
        pb::DType::DtypeUnspecified => bail!("dtype is unspecified"),
    })
}

fn to_proto_dtype(dtype: DType) -> pb::DType {
    match dtype {
        DType::F32 => pb::DType::F32,
        DType::F16 => pb::DType::F16,
        DType::I64 => pb::DType::I64,
        DType::I32 => pb::DType::I32,
        DType::U8 => pb::DType::U8,
    }
}
