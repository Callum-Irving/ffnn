use super::layer::Layer;
use super::Float;

use ndarray::Array1;
use ndarray::Array2;

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
    pub fn predict(&self, inputs: &Array1<Float>) -> Array1<Float> {
        assert_eq!(self.num_inputs, inputs.len());

        let mut last = inputs.clone();
        for layer in self.layers.iter() {
            last = layer.eval(&last);
        }
        last
    }

    /// Run forward propagation on multiple samples at once. Each row of inputs is a sample.
    /// Outputs one row per sample.
    fn predict_many(&self, inputs: &Array2<Float>) -> Array2<Float> {
        assert_eq!(self.num_inputs, inputs.ncols());

        // Transpose inputs
        let mut last = inputs.clone().reversed_axes();
        for layer in self.layers.iter() {
            last = layer.eval_many(&last);
            assert_eq!(last.ncols(), inputs.nrows());
        }
        last.reversed_axes()
    }

    /// Train for one epoch
    fn train_batch(&mut self, inputs: &Array2<Float>, targets: &Array2<Float>) {
        // Do forward prop, saving activations
        let mut sums = vec![];
        let mut activations = vec![inputs.clone().reversed_axes()];

        for layer in self.layers.iter() {
            let (z, a) = layer.eval_many_with_sum(activations.last().unwrap());
            sums.push(z);
            activations.push(a);
        }

        // Compute deltas
        let mut deltas = vec![];

        // Compute output deltas
        let output_deltas = targets - activations.last().unwrap();
        let _error: Float = (&output_deltas * &output_deltas).iter().sum();
        deltas.push(
            output_deltas
                * self
                    .layers
                    .last()
                    .unwrap()
                    .activation
                    .derive_2d(sums.last().unwrap()),
        );

        // Compute deltas for hidden layers
        for i in (0..self.layers.len() - 1).rev() {
            let delta_pullback = self.layers[i + 1]
                .weights
                .view()
                .reversed_axes()
                .dot(deltas.last().unwrap());
            let delta = delta_pullback * self.layers[i].activation.derive_2d(&sums[i]);
            deltas.push(delta);
        }
    }

    /// Do batched gradient descent by backprop.
    pub fn train(&mut self, dataset: Array2<Float>) {
        // Number of columns in dataset should match length of inputs
        assert_eq!(self.num_inputs, dataset.ncols());

        todo!();
    }

    /// Stochastic gradient descent.
    pub fn sgd(&mut self, _inputs: Array1<Float>, _targets: Array1<Float>, _lr: Float) {
        todo!()
        // use super::losses::MSE;

        // // Do forward prop and save activations
        // let mut activations: Vec<DVector<Float>> = vec![inputs.clone()];
        // for (i, layer) in self.layers.iter().enumerate() {
        //     activations.push(layer.eval(&activations[i]));
        // }

        // let err = MSE.compute_loss(activations.last().unwrap(), &targets);

        // // Dimsensionality mismatch here
        // let d_L = err.transpose() * &activations[activations.len() - 2];

        // let l = activations.len() - 3;
        // // Next hidden deltas
        // let d_1 = d_L * &self.layers[l].weights * self.layers[l].activation.derive(&activations[l]);
        // let grads = d_L.transpose() * &activations[l];

        // println!("inputs: {}", inputs);
        // println!("outputs: {}", activations.last().unwrap());
        // println!("targets: {}", targets);
        // println!("mse: {}", err);

        // todo!()

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

    /// Optimize parameters using the Adam optimization algorithm.
    pub fn adam(&mut self, _dataset: Array2<Float>) {
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
    use ndarray::prelude::*;

    use crate::activations::SIGMOID;

    use super::super::activations::RELU;
    use super::super::builder::NetBuilder;

    #[test]
    fn create() {
        NetBuilder::new(3).layer(2, RELU).layer(5, RELU).init();
    }

    #[test]
    fn predict() {
        let net = NetBuilder::new(3).layer(2, RELU).layer(5, RELU).init();
        let prediction = net.predict(&array![0.0, 0.0, 0.0]);
        assert!(prediction.len() == 5, "Output has correct dimensions");
    }

    #[test]
    fn predict_many() {
        let net = NetBuilder::new(3).layer(2, RELU).layer(5, RELU).init();
        let inputs = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let predictions = net.predict_many(&inputs);
        assert!(
            predictions.nrows() == 2,
            "Output has correct number of rows"
        );
        assert!(
            predictions.ncols() == 5,
            "Output has correct number of colums"
        );
    }

    #[test]
    fn train() {
        let mut net = NetBuilder::new(2).layer(2, RELU).layer(2, SIGMOID).init();
        net.train_batch(
            &array![[10.0, 50.0], [-20.0, 5.0]],
            &array![[1.2, 1.1], [1.4, 5.3]],
        );

        // TODO: Add gradient checking
    }
}
