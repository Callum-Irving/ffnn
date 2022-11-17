use super::activations::Activation;
use super::Float;

use rand::prelude::*;
use rand_distr::Normal;
use std::f32::consts::SQRT_2;

use ndarray::prelude::*;

pub struct Layer {
    pub weights: Array2<Float>,
    pub biases: Array1<Float>,
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with all weights set to zero.
    pub fn new(nodes: usize, inputs: usize, activation: Activation) -> Self {
        let weights = Array2::<Float>::zeros((nodes, inputs));
        let biases = Array1::<Float>::zeros(nodes);

        // let mut rng = thread_rng();
        // let norm = Normal::new(0.0, SQRT_2 / (inputs as Float).sqrt()).unwrap();
        // weights = weights.map(|_| rng.sample(norm));
        // biases = biases.map(|_| rng.sample(norm));

        Layer {
            weights,
            biases,
            activation,
        }
    }

    /// Initialize weights using He Weight Initialization.
    ///
    /// TODO: Add multiple inititalization methods
    pub fn random_init(&mut self) {
        let mut rng = thread_rng();
        let norm = Normal::new(0.0, SQRT_2 / (self.weights.nrows() as Float).sqrt()).unwrap();
        self.weights = self.weights.map(|_| rng.sample(norm));
        self.biases = self.biases.map(|_| rng.sample(norm));
    }

    /// Compute outputs for a vector of inputs.
    pub fn eval(&self, inputs: &Array1<Float>) -> Array1<Float> {
        // Check dimensionality
        assert_eq!(self.weights.ncols(), inputs.len());

        let mut out = self.weights.dot(inputs) + &self.biases;

        // Apply the activation function
        let view = out.view_mut();
        self.activation.apply(view);

        out
    }

    /// Compute outputs for multiple input vectors at once.
    pub fn eval_many(&self, inputs: &Array2<Float>) -> Array2<Float> {
        let mut out = self.weights.dot(inputs) + self.biases.slice(s![.., NewAxis]);

        // For each column, apply the activation function
        out.axis_iter_mut(Axis(1))
            .for_each(|row| self.activation.apply(row));

        out
    }

    /// The number of nodes in the layer.
    pub fn len(&self) -> usize {
        self.weights.nrows()
    }
}
