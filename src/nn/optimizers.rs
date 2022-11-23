//! Optimizers.

use crate::nn::layer::Layer;
use crate::Network;
use ndarray::prelude::*;

/// A trait implemented by optimizers such as SGD and Adam.
pub trait Optimizer {
    /// Do one iteration.
    fn step(&mut self, net: &mut Vec<Layer>, inputs: &Array2<f32>, targets: &Array2<f32>);
}

pub struct Adam {
    lr: f32,
    b1: f32,
    b2: f32,
    epsilon: f32,
    vdw: Vec<Array2<f32>>,
    sdw: Vec<Array2<f32>>,
    vdb: Vec<Array1<f32>>,
    sdb: Vec<Array1<f32>>,
    epoch: usize,
}

impl Adam {
    pub fn new(layers: &[Layer]) -> Self {
        let mut vdw = vec![];
        let mut sdw = vec![];
        let mut vdb = vec![];
        let mut sdb = vec![];

        for layer in layers.iter() {
            vdw.push(Array::zeros(layer.weights.raw_dim()));
            sdw.push(Array::zeros(layer.weights.raw_dim()));
            vdb.push(Array::zeros(layer.biases.raw_dim()));
            sdb.push(Array::zeros(layer.biases.raw_dim()));
        }

        Self {
            lr: 0.001,
            b1: 0.9,
            b2: 0.999,
            epsilon: 1e-08,
            vdw,
            sdw,
            vdb,
            sdb,
            epoch: 1,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, layers: &mut Vec<Layer>, inputs: &Array2<f32>, targets: &Array2<f32>) {
        // Do forward prop, saving activations
        let mut sums = vec![];
        let mut activations = vec![inputs.clone().reversed_axes()];

        for layer in layers.iter() {
            let (z, a) = layer.eval_many_with_sum(activations.last().unwrap());
            sums.push(z);
            activations.push(a);
        }

        // Compute deltas
        let mut deltas = vec![];

        // Compute output deltas
        let output_deltas = activations.last().unwrap() - &targets.view().reversed_axes();

        deltas.push(
            output_deltas
                * layers
                    .last()
                    .unwrap()
                    .activation
                    .derive_2d(sums.last().unwrap()),
        );

        // Compute deltas for hidden layers
        for i in (0..layers.len() - 1).rev() {
            let delta_pullback = layers[i + 1]
                .weights
                .view()
                .reversed_axes()
                .dot(deltas.last().unwrap());
            let delta = delta_pullback * layers[i].activation.derive_2d(&sums[i]);
            deltas.push(delta);
        }

        // Compute weight deltas
        // TODO: Move into hidden layer deltas step
        for i in 0..layers.len() {
            let deltas_layer = &deltas[layers.len() - i - 1];
            let prev_outputs = &activations[i];

            let first_factor = prev_outputs
                .slice(s![NewAxis, .., ..])
                .permuted_axes([2, 0, 1]);
            let second_factor = deltas_layer
                .slice(s![NewAxis, .., ..])
                .permuted_axes([2, 1, 0]);
            let product = &first_factor * &second_factor;
            let weight_deltas = product.sum_axis(Axis(0)) / inputs.nrows() as f32;
            let bias_deltas = deltas_layer.view().sum_axis(Axis(1)) / inputs.nrows() as f32;

            // Update vdw, sdw, vdb, sdb
            self.vdw[i] = self.b1 * &self.vdw[i] + (1.0 - self.b1) * &weight_deltas;
            self.sdw[i] =
                self.b2 * &self.sdw[i] + (1.0 - self.b2) * weight_deltas.mapv(|e| e.powi(2));

            self.vdb[i] = self.b1 * &self.vdb[i] + (1.0 - self.b1) * &bias_deltas;
            self.sdb[i] =
                self.b2 * &self.sdb[i] + (1.0 - self.b2) * bias_deltas.mapv(|e| e.powi(2));

            // Bias-correct them
            let vdw_corrected = &self.vdw[i] / (1.0 - self.b1.powi(self.epoch as i32));
            let sdw_corrected = &self.sdw[i] / (1.0 - self.b2.powi(self.epoch as i32));
            let vdb_corrected = &self.vdb[i] / (1.0 - self.b1.powi(self.epoch as i32));
            let sdb_corrected = &self.sdb[i] / (1.0 - self.b1.powi(self.epoch as i32));

            // Update weights and biases
            layers[i].weights -=
                &(self.lr * (vdw_corrected / (sdw_corrected.mapv(f32::sqrt) + self.epsilon)));
            layers[i].biases -=
                &(self.lr * (vdb_corrected / (sdb_corrected.mapv(f32::sqrt) + self.epsilon)));
        }

        self.epoch += 1;
    }
}
