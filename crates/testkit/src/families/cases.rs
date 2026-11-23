//! A case is an op with the element type, values and shape kind of each
//! operand; `graphs` draws single-node graphs and inputs from a list.

use super::values::{Domain, operand_dims, output_dims, shape, tensor};
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{DType, HostTensor};
use ptxc_graph::{Graph, GraphBuilder, Layout, Op, TensorType, TypeRule, ValueName};

/// How an operand's shape relates to the output's.
#[derive(Clone, Copy, Debug)]
pub enum Extent {
    /// The output's shape.
    Full,
    /// A shape that broadcasts to the output's.
    Broadcast,
    /// Rank 0.
    Scalar,
}

/// One operand of a case.
#[derive(Clone, Copy, Debug)]
pub struct OperandSpec {
    /// The element type.
    pub dtype: DType,
    /// The values it takes.
    pub domain: Domain,
    /// Its shape against the output's.
    pub extent: Extent,
}

impl OperandSpec {
    /// Any values of `dtype`, in a shape that broadcasts to the output.
    pub const fn any(dtype: DType) -> Self {
        Self { dtype, domain: Domain::Any, extent: Extent::Broadcast }
    }

    /// The same operand with shape `extent`.
    #[must_use]
    pub const fn with_extent(self, extent: Extent) -> Self {
        Self { extent, ..self }
    }

    /// The same operand with values from `domain`.
    #[must_use]
    pub const fn with_domain(self, domain: Domain) -> Self {
        Self { domain, ..self }
    }

    fn sample_type(self) -> TensorType {
        let dims: &[u64] = if matches!(self.extent, Extent::Scalar) { &[] } else { &[1] };
        TensorType::new(self.dtype, shape(dims), Layout::RowMajor)
    }

    fn dims(self, output: Vec<u64>) -> BoxedStrategy<Vec<u64>> {
        match self.extent {
            Extent::Full => Just(output).boxed(),
            Extent::Broadcast => operand_dims(output).boxed(),
            Extent::Scalar => Just(Vec::new()).boxed(),
        }
    }
}

#[derive(Clone, Debug)]
/// An op and its operands.
pub struct Case {
    /// The op.
    pub op: Op,
    /// Its operands, in order.
    pub operands: Vec<OperandSpec>,
}

impl Case {
    /// `op` on broadcasting operands of `dtypes` with any values.
    pub fn new(op: Op, dtypes: &[DType]) -> Self {
        Self { op, operands: dtypes.iter().map(|&dtype| OperandSpec::any(dtype)).collect() }
    }

    fn is_admitted(&self) -> bool {
        let types: Vec<TensorType> = self.operands.iter().map(|spec| spec.sample_type()).collect();
        self.op.infer(&types.iter().collect::<Vec<_>>()).is_ok()
    }
}

/// The cases the op's type rule accepts, of which there must be some.
pub fn admitted(cases: impl IntoIterator<Item = Case>) -> Vec<Case> {
    let admitted: Vec<Case> = cases.into_iter().filter(Case::is_admitted).collect();
    assert!(!admitted.is_empty());
    admitted
}

/// A graph of one node of a case drawn from `cases`, and inputs for it.
pub fn graphs(cases: Vec<Case>) -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    select(cases)
        .prop_flat_map(|case| {
            output_dims().prop_flat_map(move |output| {
                let inputs: Vec<BoxedStrategy<HostTensor>> = case
                    .operands
                    .iter()
                    .map(|&spec| {
                        spec.dims(output.clone())
                            .prop_flat_map(move |dims| tensor(spec.dtype, dims, spec.domain))
                            .boxed()
                    })
                    .collect();
                (Just(case.op.clone()), inputs)
            })
        })
        .prop_map(|(op, inputs)| (single_node(op, &inputs), inputs))
}

/// A graph whose one node is `op` on inputs typed as `inputs`, the inputs
/// named `x0`, `x1`, ... and the outputs `y0`, `y1`, ...
pub fn single_node(op: Op, inputs: &[HostTensor]) -> Graph {
    let mut builder = GraphBuilder::new();
    let operands: Vec<_> = inputs
        .iter()
        .map(|input| {
            builder.add_input(TensorType::new(
                input.dtype(),
                input.shape().clone(),
                Layout::RowMajor,
            ))
        })
        .collect();
    for (n, &operand) in operands.iter().enumerate() {
        builder.set_name(operand, ValueName::new(format!("x{n}")));
    }
    let outputs = builder.add_node(op, operands).unwrap();
    for (n, &output) in outputs.iter().enumerate() {
        builder.set_name(output, ValueName::new(format!("y{n}")));
    }
    builder.finish(outputs).unwrap()
}
