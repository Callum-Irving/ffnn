use super::layer::Layer;
use super::Float;

use nalgebra::{DMatrix, DVector};

/// Basic feedforward neural network.
///
/// It is generally constructed using a [`NetBuilder`](crate::NetBuilder).
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
            layer.random_init();
        }
    }

    /// Do forward propagation.
    pub fn predict(&self, inputs: DVector<Float>) -> DVector<Float> {
        assert_eq!(self.num_inputs, inputs.len());

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

    /// Stochastic gradient descent.
    pub fn sgd(&mut self, inputs: DVector<Float>, targets: DVector<Float>) {
        use super::losses::MSE;

        // TODO: Save all intermediate activations
        let mut outputs = self.predict(inputs);

        let output_errors = MSE.compute_loss(&outputs, &targets);
        for layer in self.layers.iter_mut().rev() {
            let grads = layer.activation.as_ref().unwrap().derive(&outputs);
            let s2 = grads * output_errors;
            // let s3 = grads * "transpose of last layer activations";
            let fin = s2 * 0.3; // Learning rate
            // add fin to layer.weights
            layer.weights += fin;
            outputs = dvector![];
        }
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
    use nalgebra::dvector;

    use super::super::activations::RELU;
    use super::super::builder::NetBuilder;

    #[test]
    fn create() {
        let net = NetBuilder::new(3)
            .layer(2, None)
            .layer(5, Some(RELU))
            .init();

        net.predict(dvector![0.0, 0.0, 0.0]);
    }
}
