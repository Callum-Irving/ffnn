use super::activations::Activation;
use super::Float;

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::Normal;
use std::f32::consts::SQRT_2;

pub struct Layer {
    pub weights: DMatrix<Float>,
    activation: Option<Activation>,
}

impl Layer {
    /// Create a new layer with all weights set to zero.
    pub fn new(nodes: usize, inputs: usize, activation: Option<Activation>) -> Self {
        // Create matrix with <nodes> rows and <inputs> + 1 columns
        // The + 1 is for the bias weight
        let weights = DMatrix::<Float>::zeros(nodes, inputs + 1);

        Layer {
            weights,
            activation,
        }
    }

    /// Initialize weights using He Weight Initialization.
    pub fn random_init(&mut self) {
        let mut rng = thread_rng();
        let norm = Normal::new(0.0, SQRT_2 / (self.weights.nrows() as f32).sqrt()).unwrap();
        self.weights = self.weights.map(|_| rng.sample(norm));
    }

    /// Compute outputs for a vector of inputs.
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

    /// The number of nodes in the layer.
    pub fn len(&self) -> usize {
        self.weights.nrows()
    }
}
