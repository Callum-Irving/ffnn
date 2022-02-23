use nalgebra::{DMatrix, DVector};

pub struct Layer {
    weights: DMatrix<f32>,
    activation: Option<fn(f32) -> f32>,
}

impl Layer {
    pub fn new(nodes: usize, inputs: usize, activation: Option<fn(f32) -> f32>) -> Self {
        // Create matrix with <nodes> columns and <inputs> rows
        let weights = DMatrix::<f32>::zeros(inputs, nodes);

        Layer {
            weights,
            activation,
        }
    }

    pub fn eval(&self, inputs: DVector<f32>) -> DVector<f32> {
        assert_eq!(self.weights.ncols(), inputs.len());
        let mut out = DVector::zeros(inputs.len());
        self.weights.mul_to(&inputs, &mut out);
        out
    }

    pub fn len(&self) -> usize {
        self.weights.ncols()
    }
}
