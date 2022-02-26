use nalgebra::{DMatrix, DVector, dvector};
use rand::prelude::*;

pub struct Layer {
    pub weights: DMatrix<f32>,
    activation: Option<fn(f32) -> f32>,
}

impl Layer {
    pub fn new(nodes: usize, inputs: usize, activation: Option<fn(f32) -> f32>) -> Self {
        // Create matrix with <nodes> rows and <inputs> + 1 columns
        // The + 1 is for the bias weight
        let weights = DMatrix::<f32>::zeros(nodes, inputs + 1);

        Layer {
            weights,
            activation,
        }
    }

    pub fn random_init(&mut self, min: f32, max: f32) {
        for weight in self.weights.iter_mut() {
            *weight = thread_rng().gen_range(min..max);
        }
    }

    pub fn eval(&self, inputs: DVector<f32>) -> DVector<f32> {
        assert_eq!(self.weights.ncols() - 1, inputs.len());

        // Append 1 to the end of the input vector
        let inputs = inputs.push(1.0);

        let mut out = DVector::zeros(self.weights.nrows());
        self.weights.mul_to(&inputs, &mut out);

        if let Some(activation) = self.activation {
            out.map(|result| activation(result))
        } else {
            out
        }
    }

    pub fn len(&self) -> usize {
        self.weights.nrows()
    }
}
