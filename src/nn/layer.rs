use nalgebra::DMatrix;

pub struct Layer {
    weights: DMatrix<f32>,
    activation: fn(f32) -> f32,
}

impl Layer {
    pub fn new(nodes: usize, activation: Option<fn(f32) -> f32>) -> Self {
        todo!();
    }

    pub fn eval(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(self.weights.ncols(), inputs.len());
        todo!();
    }

    pub fn len(&self) -> usize {
        self.weights.ncols()
    }
}
