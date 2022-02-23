use nalgebra::DMatrix;

pub struct Layer {
    weights: DMatrix<f32>,
    activation: Option<fn(f32) -> f32>,
}

impl Layer {
    pub fn new(nodes: usize, inputs: usize, activation: Option<fn(f32) -> f32>) -> Self {
        // Create matrix with <nodes> columns and <inputs> rows
        let weights = DMatrix::<f32>::zeros(inputs, nodes);

        Layer {
            weights,
            activation,
        }
    }

    pub fn eval(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(self.weights.ncols(), inputs.len());
        todo!();
    }

    pub fn len(&self) -> usize {
        self.weights.ncols()
    }
}
