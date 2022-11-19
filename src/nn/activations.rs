//! Activation functions.
//!
//! Predefined functions are Sigmoid, ReLU, and Softmax. Users can create their own activation
//! functions using the [`Activation`] struct.

use super::Float;

use ndarray::prelude::*;

// TODO: Make some sort of macro for defining activation functions

/// A struct that represents an activation function.
///
/// Contains a function for forward pass and backward pass.
pub struct Activation {
    /// The function used for forward propagation.
    apply: fn(ArrayViewMut1<Float>),

    /// The function used for backpropagation.
    derive: fn(ArrayViewMut1<Float>),
}

impl Activation {
    fn apply_mut(&self, inputs: ArrayViewMut1<Float>) {
        (self.apply)(inputs);
    }

    /// Apply activation function for forward propagation.
    pub fn apply(&self, mut inputs: Array1<Float>) -> Array1<Float> {
        let view = inputs.view_mut();
        self.apply_mut(view);
        inputs
    }

    pub fn apply_2d(&self, mut inputs: Array2<Float>) -> Array2<Float> {
        // Apply to each column
        inputs
            .axis_iter_mut(Axis(1))
            .for_each(|col| self.apply_mut(col));

        inputs
    }

    fn derive_mut(&self, inputs: ArrayViewMut1<Float>) {
        (self.derive)(inputs)
    }

    /// Get gradients for set of inputs.
    pub fn derive(&self, inputs: &Array1<Float>) -> Array1<Float> {
        let mut res = inputs.clone();
        let view = res.view_mut();
        self.derive_mut(view);
        res
    }

    pub fn derive_2d(&self, inputs: &Array2<Float>) -> Array2<Float> {
        let mut res = inputs.clone();
        res.axis_iter_mut(Axis(1))
            .for_each(|col| self.derive_mut(col));

        res
    }
}

/// The logistic function.
///
/// TODO: describe function
pub const SIGMOID: Activation = Activation {
    apply: |mut z| z.iter_mut().for_each(|f| *f = sigmoid(*f)),
    derive: |mut z| z.iter_mut().for_each(|f| *f = d_sigmoid(*f)),
};

/// The ReLU activation function.
///
/// TODO: Describe relu
pub const RELU: Activation = Activation {
    apply: |mut z| z.iter_mut().for_each(|f| *f = relu(*f)),
    derive: |mut z| z.iter_mut().for_each(|f| *f = d_relu(*f)),
};

/// The Softmax activation function.
///
/// TODO: Add description of softmax.
pub const SOFTMAX: Activation = Activation {
    apply: |_| todo!(),
    derive: |_| todo!(),
};

fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

fn d_sigmoid(x: Float) -> Float {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn relu(x: Float) -> Float {
    let zero: Float = 0.0;
    zero.max(x)
}

fn d_relu(x: Float) -> Float {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: Float = 0.000001;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < EPSILON);
        assert!((sigmoid(2.0) - 0.880797).abs() < EPSILON);
        assert!((sigmoid(-2.0) - 0.119202).abs() < EPSILON);

        assert!((d_sigmoid(0.0) - 0.25).abs() < EPSILON);
        assert!((d_sigmoid(2.0) - 0.104993).abs() < EPSILON);
        assert!((d_sigmoid(-2.0) - 0.104993).abs() < EPSILON);
    }

    #[test]
    fn test_relu() {
        assert!((relu(2.0) - 2.0).abs() < EPSILON);
        assert!((relu(0.0)).abs() < EPSILON);
        assert!((relu(-2.0)).abs() < EPSILON);

        assert!((d_relu(2.0) - 1.0).abs() < EPSILON);
        assert!((d_relu(0.0)).abs() < EPSILON);
        assert!((d_relu(-1.0)).abs() < EPSILON);
    }
}
