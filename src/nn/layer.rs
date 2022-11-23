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

        // Compute weighted sum
        let out = self.weights.dot(inputs) + &self.biases;

        // Apply activation
        let out = self.activation.apply(out);

        out
    }

    pub fn eval_with_sum(&self, inputs: &Array1<Float>) -> (Array1<Float>, Array1<Float>) {
        let z = self.weights.dot(inputs) + &self.biases;
        let a = self.activation.apply(z.clone());
        (z, a)
    }

    /// Compute outputs for multiple input vectors at once.
    pub fn eval_many(&self, inputs: &Array2<Float>) -> Array2<Float> {
        let out = self.weights.dot(inputs) + self.biases.slice(s![.., NewAxis]);

        // Apply activation to each column
        let out = self.activation.apply_2d(out);

        out
    }

    pub fn eval_many_with_sum(&self, inputs: &Array2<Float>) -> (Array2<Float>, Array2<Float>) {
        let z = self.weights.dot(inputs) + self.biases.slice(s![.., NewAxis]);
        let a = self.activation.apply_2d(z.clone());

        (z, a)
    }

    /// The number of nodes in the layer.
    pub fn len(&self) -> usize {
        self.weights.nrows()
    }
}

#[cfg(test)]
mod tests {
    use super::super::activations::*;
    use super::*;

    const EPSILON: Float = 0.000001;

    #[test]
    fn test_eval_1d() {
        let layer = Layer::new(2, 1, SIGMOID);
        assert!((&layer.eval(&array![111.0]) - &array![0.5, 0.5]).sum() < 2.0 * EPSILON);
    }

    #[test]
    fn test_eval_2d() {
        let layer = Layer::new(2, 1, SIGMOID);
        let inputs = array![[1.0, 1.0]];
        let (z, a) = layer.eval_many_with_sum(&inputs);
        assert!((&z - &array![[0.0, 0.0]]).sum() < 2.0 * EPSILON);
        assert!((&a - &array![[0.5, 0.5]]).sum() < 2.0 * EPSILON);
    }
}
