use super::layer::Layer;
use super::Float;

use nalgebra::{DMatrix, DVector};

/// Basic feedforward neural network
pub struct Network {
    num_inputs: usize,
    layers: Vec<Layer>,
}

impl Network {
    /// Create a new network from number of inputs and layers.
    pub fn new(inputs: usize, layers: Vec<Layer>) -> Self {
        Network {
            num_inputs: inputs,
            layers,
        }
    }

    /// Perform some sort of initialization.
    pub fn init(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.random_init(-1.0, 1.0);
        }
    }

    /// Do forward propagation.
    pub fn predict(&self, inputs: Vec<Float>) -> DVector<Float> {
        assert_eq!(self.num_inputs, inputs.len());

        let inputs = DVector::from_vec(inputs);
        let mut last = inputs;
        for layer in self.layers.iter() {
            last = layer.eval(last);
        }
        last
    }

    /// Do batched gradient descent by backprop.
    pub fn train(&mut self, dataset: DMatrix<Float>) {
        // Number of columns in dataset should match length of inputs
        assert_eq!(self.num_inputs, dataset.ncols());

        todo!();
    }

    /// Print a summary of all the weights in the neural network.
    pub fn print(&self) {
        for layer in self.layers.iter() {
            println!("{}", layer.weights);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::activations::RELU;
    use super::super::builder::NetBuilder;

    #[test]
    fn create() {
        let net = NetBuilder::new(3)
            .layer(2, None)
            .layer(5, Some(RELU))
            .init();

        net.predict(vec![0.0, 0.0, 0.0]);
    }
}
