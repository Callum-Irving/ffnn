//! Activation functions.
//!
//! Predefined functions are Sigmoid, ReLU, and Softmax. Users can create their own activation
//! functions using the [`Activation`] struct.

use super::Float;

use nalgebra::DVector;

// TODO: Make some sort of macro for defining activation functions

/// A struct that represents an activation function.
///
/// Contains a function for forward pass and backward pass.
pub struct Activation {
    /// The function used for forward propagation.
    apply: fn(DVector<Float>) -> DVector<Float>,

    /// The function used for backpropagation.
    derive: fn(&DVector<Float>) -> DVector<Float>,
}

impl Activation {
    /// Apply activation function for forward propagation.
    pub fn apply(&self, inputs: DVector<Float>) -> DVector<Float> {
        (self.apply)(inputs)
    }

    /// Get gradients for set of inputs.
    pub fn derive(&self, inputs: &DVector<Float>) -> DVector<Float> {
        (self.derive)(inputs)
    }
}

/// The logistic function.
///
/// TODO: describe function
pub const SIGMOID: Activation = Activation {
    apply: |z| z.map(sigmoid),
    derive: |z| z.map(d_sigmoid),
};

/// The ReLU activation function.
///
/// TODO: Describe relu
pub const RELU: Activation = Activation {
    apply: |z| z.map(relu),
    derive: |z| z.map(d_relu),
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
