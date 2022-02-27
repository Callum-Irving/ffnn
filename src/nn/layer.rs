use super::activations::Activation;
use super::Float;

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;

pub struct Layer {
    pub weights: DMatrix<Float>,
    activation: Option<Activation>,
}

impl Layer {
    pub fn new(nodes: usize, inputs: usize, activation: Option<Activation>) -> Self {
        // Create matrix with <nodes> rows and <inputs> + 1 columns
        // The + 1 is for the bias weight
        let weights = DMatrix::<Float>::zeros(nodes, inputs + 1);

        Layer {
            weights,
            activation,
        }
    }

    pub fn random_init(&mut self, min: Float, max: Float) {
        for weight in self.weights.iter_mut() {
            *weight = thread_rng().gen_range(min..max);
        }
    }

    pub fn eval(&self, inputs: DVector<Float>) -> DVector<Float> {
        assert_eq!(self.weights.ncols() - 1, inputs.len());

        // Append 1 to the end of the input vector
        let inputs = inputs.push(1.0);

        let mut out = DVector::zeros(self.weights.nrows());
        self.weights.mul_to(&inputs, &mut out);

        if let Some(activation) = &self.activation {
            (activation.apply)(out)
        } else {
            out
        }
    }

    pub fn len(&self) -> usize {
        self.weights.nrows()
    }
}
