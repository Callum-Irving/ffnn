//! Loss functions.

use super::Float;

use nalgebra::DVector;

/// A struct that contains a loss function.
pub struct Loss {
    /// The loss function.
    compute: fn(&DVector<Float>, &DVector<Float>) -> Float,
}

impl Loss {
    /// Calls the loss function contained in `self`.
    pub fn compute_loss(&self, outputs: &DVector<Float>, targets: &DVector<Float>) -> Float {
        (self.compute)(outputs, targets)
    }
}

/// Mean squared error.
///
/// TODO: Describe it.
pub const MSE: Loss = Loss { compute: mse_loss };

fn mse_loss(outputs: &DVector<Float>, targets: &DVector<Float>) -> Float {
    (outputs - targets).map(|x| x * x).sum() / 2.0
}
