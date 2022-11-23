use super::layer::Layer;
use super::Float;
use crate::optimizers::Adam;
use crate::optimizers::Optimizer;

use ndarray::prelude::*;

/// Basic feedforward neural network.
///
/// It is generally constructed using a [`NetBuilder`](crate::NetBuilder).
pub struct Network {
    pub num_inputs: usize,
    pub layers: Vec<Layer>,
    optimizer: Adam,
}

impl Network {
    /// Create a new network from number of inputs and layers.
    pub fn new(inputs: usize, layers: Vec<Layer>, optimizer: Adam) -> Self {
        Network {
            num_inputs: inputs,
            layers,
            optimizer,
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
    // Outputs one row per sample.
    pub fn predict_many(&self, inputs: &Array2<Float>) -> Array2<Float> {
        assert_eq!(self.num_inputs, inputs.ncols());

        // Transpose inputs
        let mut last = inputs.clone().reversed_axes();
        for layer in self.layers.iter() {
            last = layer.eval_many(&last);
            assert_eq!(last.ncols(), inputs.nrows());
        }
        last.reversed_axes()
    }

    pub fn num_outputs(&self) -> usize {
        self.layers.last().unwrap().weights.nrows()
    }

    /// Train for one epoch on provided mini-batch `inputs`.
    pub fn train_batch(
        &mut self,
        inputs: &Array2<Float>,
        targets: &Array2<Float>,
        lr: Float,
    ) -> f32 {
        // Check dimensions
        debug_assert_eq!(self.num_inputs, inputs.ncols());
        debug_assert_eq!(self.num_outputs(), targets.ncols());
        debug_assert_eq!(inputs.nrows(), targets.nrows());

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
        let output_deltas = activations.last().unwrap() - &targets.view().reversed_axes();

        // Compute MSE for batch:
        let error: Float = (&output_deltas * &output_deltas).iter().sum::<Float>()
            / (2.0 * inputs.nrows() as Float);

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

        // Compute weight deltas
        // TODO: Move into hidden layer deltas step
        for i in 0..self.layers.len() {
            let deltas_layer = &deltas[self.layers.len() - i - 1];
            let prev_outputs = &activations[i];

            let first_factor = prev_outputs
                .slice(s![NewAxis, .., ..])
                .permuted_axes([2, 0, 1]);
            let second_factor = deltas_layer
                .slice(s![NewAxis, .., ..])
                .permuted_axes([2, 1, 0]);
            let product = &first_factor * &second_factor;
            let weight_deltas = product.sum_axis(Axis(0)) / inputs.nrows() as Float;
            let bias_deltas = deltas_layer.view().sum_axis(Axis(1)) / inputs.nrows() as Float;

            // Checking dimensionality of deltas:
            debug_assert_eq!(weight_deltas.nrows(), self.layers[i].weights.nrows());
            debug_assert_eq!(weight_deltas.ncols(), self.layers[i].weights.ncols());
            debug_assert_eq!(bias_deltas.len(), self.layers[i].biases.len());

            self.layers[i].weights -= &(weight_deltas * lr);
            self.layers[i].biases -= &(bias_deltas * lr);
        }

        error
    }

    /// Return the mean squared error of `targets` and the prediction of the network given `inputs`.
    pub fn mse(&self, inputs: &Array2<Float>, targets: &Array2<Float>) -> Float {
        let outputs = self.predict_many(&inputs);
        let output_deltas = targets - outputs;
        let error: Float = (&output_deltas * &output_deltas).iter().sum::<Float>()
            / (2.0 * inputs.nrows() as Float);

        error
    }

    /// Do batched gradient descent by backprop.
    pub fn train(
        &mut self,
        inputs: &Array2<Float>,
        targets: &Array2<Float>,
        epochs: usize,
        batch_size: usize,
        lr: f32,
    ) -> Vec<f32> {
        // Check dimensions
        assert_eq!(self.num_inputs, inputs.ncols());
        assert_eq!(self.num_outputs(), targets.ncols());
        assert_eq!(inputs.nrows(), targets.nrows());

        let mut errors = Vec::with_capacity(epochs);
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        let mut rng = thread_rng();

        for _ in 0..epochs {
            // Split into batches
            // Don't rearrange inputs, just use views somehow
            // shuffle(range(inputs.ncols())) and then sample in batches of batch_size
            let mut indices = (0..inputs.ncols()).collect::<Vec<usize>>();
            indices.shuffle(&mut rng);
            let iter = indices.chunks(batch_size);

            for batch_indices in iter {
                // This select thing clones but I guess that is needed anyways :(
                let batch_inputs = inputs.select(Axis(0), batch_indices);
                let batch_targets = targets.select(Axis(0), batch_indices);
                // self.train_batch(&batch_inputs, &batch_targets, lr);

                self.optimizer
                    .step(&mut self.layers, &batch_inputs, &batch_targets);
            }

            // Compute MSE for epoch
            errors.push(self.mse(inputs, targets));
        }

        errors
    }

    /// Stochastic gradient descent.
    pub fn sgd(&mut self, _inputs: Array1<Float>, _targets: Array1<Float>, _lr: Float) {
        todo!()
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
    use super::Network;
    use crate::nn::Float;

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

    /// Perform gradient checking
    ///
    /// Note to self: This is an absolutely monstrous function. Fix it in the future pls.
    fn gradient_check(net: &mut Network, inputs: &Array2<Float>, targets: &Array2<Float>) {
        // Check dimensions
        assert_eq!(net.num_inputs, inputs.ncols());
        assert_eq!(net.num_outputs(), targets.ncols());
        assert_eq!(inputs.nrows(), targets.nrows());

        // Do forward prop, saving activations
        let mut sums = vec![];
        let mut activations = vec![inputs.clone().reversed_axes()];

        for layer in net.layers.iter() {
            let (z, a) = layer.eval_many_with_sum(activations.last().unwrap());
            sums.push(z);
            activations.push(a);
        }

        // Compute deltas
        let mut deltas = vec![];

        // Compute output deltas
        let output_deltas = activations.last().unwrap() - &targets.view().reversed_axes();

        // check output deltas

        deltas.push(
            output_deltas
                * net
                    .layers
                    .last()
                    .unwrap()
                    .activation
                    .derive_2d(sums.last().unwrap()),
        );

        // Compute deltas for hidden layers
        // for layer i:
        // z = sums[i]
        // a = activations[i+1]
        for i in (0..net.layers.len() - 1).rev() {
            let delta_pullback = net.layers[i + 1]
                .weights
                .view()
                .reversed_axes()
                .dot(deltas.last().unwrap());
            let delta = delta_pullback * net.layers[i].activation.derive_2d(&sums[i]);
            deltas.push(delta);
        }

        assert_eq!(deltas[0].ncols(), inputs.nrows());

        let mut weight_deltas = vec![];
        let mut bias_deltas = vec![];

        // Compute weight deltas
        //
        // for layer i:
        // deltas = layer_count - i - 1
        for i in 0..net.layers.len() {
            let deltas_layer = &deltas[net.layers.len() - i - 1];
            let prev_outputs = &activations[i];

            let first_factor = prev_outputs
                .slice(s![NewAxis, .., ..])
                .permuted_axes([2, 0, 1]);
            let second_factor = deltas_layer
                .slice(s![NewAxis, .., ..])
                .permuted_axes([2, 1, 0]);
            let product = &first_factor * &second_factor;
            let weight_delta = product.sum_axis(Axis(0)) / inputs.nrows() as Float;
            // TODO: Compute bias deltas
            let bias_delta = deltas_layer.view().sum_axis(Axis(1)) / inputs.nrows() as Float;
            //assert_eq!(weight_deltas.nrows(), net.layers[i].weights.nrows());
            //assert_eq!(weight_deltas.ncols(), net.layers[i].weights.ncols());
            //assert_eq!(bias_deltas.len(), net.layers[i].biases.len());

            weight_deltas.push(weight_delta);
            bias_deltas.push(bias_delta);
            //net.layers[i].weights = &net.layers[i].weights - (weight_deltas * lr);
            //net.layers[i].biases = &net.layers[i].biases - (bias_deltas * lr);
        }

        // Check gradients
        let epsilon = 10e-7_f32;
        for i_layer in (0..net.layers.len()).rev() {
            // For each weight
            for i_row in 0..net.layers[i_layer].weights.nrows() {
                for i_col in 0..net.layers[i_layer].weights.ncols() {
                    let w = net.layers[i_layer].weights[[i_row, i_col]];

                    // Compute mse with slightly less
                    net.layers[i_layer].weights[[i_row, i_col]] = w + epsilon;
                    let mse_more = net.mse(inputs, targets);

                    // Compute mse with slightly more
                    net.layers[i_layer].weights[[i_row, i_col]] = w - epsilon;
                    let mse_less = net.mse(inputs, targets);

                    let grad = (mse_more - mse_less) / (2.0 * epsilon);

                    assert!(
                        (f32::abs(grad - weight_deltas[i_layer][[i_row, i_col]])) < 10e-6,
                        "Gradient check failed at: Layer: {}, Row: {}, Col: {}. Expected: {}, Found: {}",
                        i_layer,
                        i_row,
                        i_col,
                        grad,
                        weight_deltas[i_layer][[i_row, i_col]]
                    );

                    net.layers[i_layer].weights[[i_row, i_col]] = w;
                }
            }

            // For each bias
            for i_bias in 0..net.layers[i_layer].biases.len() {
                let b = net.layers[i_layer].biases[i_bias];

                net.layers[i_layer].biases[i_bias] = b + epsilon;
                let mse_more = net.mse(inputs, targets);

                net.layers[i_layer].biases[i_bias] = b - epsilon;
                let mse_less = net.mse(inputs, targets);

                let grad = (mse_more - mse_less) / (2.0 * epsilon);
                assert!(
                    (grad - bias_deltas[i_layer][i_bias]).powi(2) < 10e-5,
                    "Gradient check failed at: Layer: {}, Bias: {}. Expected: {}, Found: {}",
                    i_layer,
                    i_bias,
                    grad,
                    bias_deltas[i_layer][i_bias]
                );

                net.layers[i_layer].biases[i_bias] = b;
            }
        }
    }

    #[test]
    fn grad_check() {
        let xor_x = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let xor_y = array![[0.05], [0.95], [0.95], [0.05]];

        //let xor_x = array![[0., 0.]];
        //let xor_y = array![[1.]];

        let mut net = NetBuilder::new(2)
            .layer(2, SIGMOID)
            .layer(1, SIGMOID)
            .init();

        for _ in 0..100 {
            net.train_batch(&xor_x, &xor_y, 0.001);
        }

        gradient_check(&mut net, &xor_x, &xor_y);
    }

    #[test]
    fn train_simple() {
        let x = array![[1.0]];
        let y = array![[0.5]];
        let mut net = NetBuilder::new(1).layer(1, SIGMOID).init();
        let e1 = net.train_batch(&x, &y, 0.1);
        for _ in 0..1000 {
            net.train_batch(&x, &y, 0.1);
        }
        let e2 = net.train_batch(&x, &y, 0.1);
        println!("{} -> {}", e1, e2);
        panic!();
    }

    #[test]
    fn train() {
        let xor_x = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let xor_y = array![[0.00], [1.00], [1.00], [0.00]];
        let mut net = NetBuilder::new(2)
            .layer(2, SIGMOID)
            .layer(1, SIGMOID)
            .init();
        let e1 = net.train_batch(&xor_x, &xor_y, 0.2);
        println!("{}", e1);
        for _ in 0..100000 {
            net.train_batch(&xor_x, &xor_y, 0.2);
        }
        let e2 = net.train_batch(&xor_x, &xor_y, 0.2);
        println!("{}", e2);

        println!("{}", net.predict(&array![0., 1.]));
        panic!();

        // TODO: Add gradient checking
    }

    #[test]
    fn train_adam() {
        let xor_x = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let xor_y = array![[0.00], [1.00], [1.00], [0.00]];
        let mut net = NetBuilder::new(2)
            .layer(2, SIGMOID)
            .layer(1, SIGMOID)
            .init();
        net.train(&xor_x, &xor_y, 100000, 4, 1000.0);
        println!("Error after: {}", net.mse(&xor_x, &xor_y));
        panic!();
    }
}
