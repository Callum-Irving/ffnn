//! Optimizers.

/// A trait implemented by optimizers such as SGD and Adam.
pub trait Optimizer {
    /// Do one iteration.
    fn step();
}

struct Adam;

impl Optimizer for Adam {
    fn step() {
        todo!();
    }
}
