//! Loss functions.

use super::Float;

use ndarray::prelude::*;

/// A struct that contains a loss function.
pub struct Loss {
    /// The loss function.
    compute: fn(&Array1<Float>, &Array1<Float>) -> Array1<Float>,
}

impl Loss {
    /// Calls the loss function contained in `self`.
    pub fn compute_loss(&self, outputs: &Array1<Float>, targets: &Array1<Float>) -> Array1<Float> {
        (self.compute)(outputs, targets)
    }
}

/// Mean squared error.
///
/// TODO: Describe it.
pub const MSE: Loss = Loss { compute: mse_loss };

fn mse_loss(outputs: &Array1<Float>, targets: &Array1<Float>) -> Array1<Float> {
    (outputs - targets).mapv(|x| x * x) / 2.0
}
