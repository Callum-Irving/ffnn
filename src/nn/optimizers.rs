//! Optimizers.

pub trait Optimizer {
    fn step();
}

struct Adam;

impl Optimizer for Adam {
    fn step() {
        todo!();
    }
}
