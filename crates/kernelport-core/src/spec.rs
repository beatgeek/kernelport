#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IOName(pub String);

#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub name: IOName,
    pub dtype: super::DType,
    pub rank: usize,
    pub dims: Vec<Option<usize>>, // None = dynamic
}

#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub max_batch: usize,
}

