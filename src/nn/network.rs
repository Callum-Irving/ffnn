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
            last = layer.eval(&last);
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
    pub fn sgd(&mut self, inputs: DVector<Float>, targets: DVector<Float>, _lr: Float) {
        use super::losses::MSE;

        // Do forward prop and save activations
        let mut activations: Vec<DVector<Float>> = vec![inputs.clone()];
        for (i, layer) in self.layers.iter().enumerate() {
            activations.push(layer.eval(&activations[i]));
        }

        let err = MSE.compute_loss(activations.last().unwrap(), &targets);

        println!("inputs: {}", inputs);
        println!("outputs: {}", activations.last().unwrap());
        println!("targets: {}", targets);
        println!("mse: {}", err);

        todo!()
        // let mut grads_last = self.layers[self.layers.len() - 1]
        //     .activation
        //     .derive(activations.last().unwrap());

        // for (i, layer) in self.layers.iter_mut().rev().skip(1).enumerate() {
        //     let grads = layer.activation.derive(&activations[i]);
        //     let s2 = grads.clone() * grads_last * lr;
        //     layer.weights += s2;
        //     grads_last = grads;
        // }

        // for (i, layer) in self.layers.iter_mut().rev().enumerate() {
        //     let grads = layer.activation.as_ref().unwrap().derive(&activations[i]);
        //     let s2 = grads * last_err * lr;
        //     layer.weights += s2;
        //     last_err = grads;
        // }
        //
        //
        //
        // for layer in self.layers.iter_mut().rev() {
        //     let grads = layer.activation.as_ref().unwrap().derive(&outputs);
        //     let s2 = grads * output_errors;
        //     // let s3 = grads * "transpose of last layer activations";
        //     let fin = s2 * 0.3; // Learning rate
        //     // add fin to layer.weights
        //     layer.weights += fin;
        //     // outputs = dvector![];
        // }
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

    use crate::activations::SIGMOID;

    use super::super::activations::RELU;
    use super::super::builder::NetBuilder;

    #[test]
    fn create() {
        let net = NetBuilder::new(3).layer(2, RELU).layer(5, RELU).init();

        net.predict(dvector![0.0, 0.0, 0.0]);
    }

    #[test]
    fn train() {
        let mut net = NetBuilder::new(3).layer(2, RELU).layer(5, SIGMOID).init();
        net.sgd(
            dvector![10.0, 50.0, -20.0],
            dvector![1.2, 1.1, 1.4, 5.3, 8.6],
            0.1,
        );
    }
}
