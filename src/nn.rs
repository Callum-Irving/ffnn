use nalgebra::DMatrix;

struct Layer {
    weights: DMatrix<f32>,
}

struct Network {
    inputs: usize,
    hidden: Vec<Layer>,
    outputs: DMatrix<f32>,
}

impl Network {
    pub fn predict(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(self.inputs, inputs.len());
        todo!();
    }

    pub fn train(&mut self, dataset: DMatrix<f32>) {
        // Number of columns in dataset should match length of inputs
        assert_eq!(dataset.ncols(), self.inputs);

        todo!();
    }
}

// Idea for builder pattern:
//
// NetBuilder::new()
//      .layer(5, Activation:Relu)
//      .layer(2, Activation::Sigmoid)
//      .outputs(5, Activation::SoftMax);
