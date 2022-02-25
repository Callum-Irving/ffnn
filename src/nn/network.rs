use super::layer::Layer;

use nalgebra::{DMatrix, DVector};

/// Basic feedforward neural network
pub struct Network {
    num_inputs: usize,
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(inputs: usize, layers: Vec<Layer>) -> Self {
        Network { num_inputs: inputs, layers }
    }

    /// Perform some sort of initialization.
    pub fn init(&mut self) {
        todo!();
    }

    /// Do forward propagation.
    pub fn predict(&self, inputs: Vec<f32>) -> DVector<f32> {
        assert_eq!(self.num_inputs, inputs.len());

        let inputs = DVector::from_vec(inputs);
        let mut last = inputs;
        for layer in self.layers.iter() {
            last = layer.eval(last);
        }
        last
    }

    /// Do batched gradient descent by backprop.
    pub fn train(&mut self, dataset: DMatrix<f32>) {
        // Number of columns in dataset should match length of inputs
        assert_eq!(self.num_inputs, dataset.ncols());

        todo!();
    }
}
