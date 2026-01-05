use bytes::Bytes;
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub enum Device {
    Cpu,
    Cuda { device_id: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I64,
    I32,
    U8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub SmallVec<[usize; 6]>);

impl Shape {
    pub fn from_slice(d: &[usize]) -> Self {
        Self(d.iter().copied().collect())
    }
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    pub fn numel(&self) -> usize {
        self.0.iter().product::<usize>().max(1)
    }
}

#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub dtype: DType,
    pub shape: Shape,
    pub device: Device,
    pub strides: Option<SmallVec<[isize; 6]>>,
}

#[derive(Clone, Debug)]
pub struct PinnedBuf {
    pub bytes: Bytes,
}

#[derive(Clone, Debug)]
pub struct CudaBuf {
    pub device_id: u32,
    pub bytes: Bytes, // placeholder (real impl = device pointer + drop)
}

/// Owns the storage for a tensor.
/// Start CPU-only; expand to pinned + cuda pools later.
#[derive(Clone, Debug)]
pub enum TensorStorage {
    CpuBytes(Bytes),
    CpuPinned(PinnedBuf),
    CudaDevice(CudaBuf),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub desc: TensorDesc,
    pub storage: TensorStorage,
    pub byte_len: usize,
}

impl Tensor {
    pub fn from_cpu_bytes(dtype: DType, shape: Shape, bytes: Bytes) -> Self {
        let byte_len = bytes.len();
        Self {
            desc: TensorDesc {
                dtype,
                shape,
                device: Device::Cpu,
                strides: None,
            },
            storage: TensorStorage::CpuBytes(bytes),
            byte_len,
        }
    }
}
