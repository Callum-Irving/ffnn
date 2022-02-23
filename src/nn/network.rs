use super::layer::Layer;

use nalgebra::DMatrix;

/// Basic feedforward neural network
pub struct Network {
    inputs: Layer,
    hidden: Vec<Layer>,
    outputs: Layer,
}

impl Network {
    pub fn new(inputs: usize, hidden: Vec<Layer>, outputs: Layer) -> Self {
        Network {
            inputs: Layer::new(inputs, 1, None),
            hidden,
            outputs,
        }
    }

    /// Perform some sort of initialization.
    pub fn init(&mut self) {
        todo!();
    }

    /// Do forward propagation.
    pub fn predict(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(self.inputs.len(), inputs.len());

        let mut last = inputs;
        for layer in self.hidden.iter() {
            last = layer.eval(last);
        }

        self.outputs.eval(last)
    }

    /// Do batched gradient descent by backprop.
    pub fn train(&mut self, dataset: DMatrix<f32>) {
        // Number of columns in dataset should match length of inputs
        assert_eq!(dataset.ncols(), self.inputs.len());

        todo!();
    }
}
