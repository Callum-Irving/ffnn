use nalgebra::DMatrix;

struct Layer {
    weights: DMatrix<f32>,
    activation: fn(f32) -> f32,
}

impl Layer {
    pub fn eval(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(self.weights.ncols(), inputs.len());
        todo!();
    }
}

struct Network {
    inputs: usize,
    hidden: Vec<Layer>,
    outputs: Layer,
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

struct NetBuilder {
    layers: Vec<Layer>,
}

impl NetBuilder {}

// TODO: Make Layer and Network generics

// TODO: Helper function that converts dataset ( Vec<Vec<f32>> ) into a DMatrix<f32?

// Idea for builder pattern:
//
// NetBuilder::new()
//      .layer(5, Activation:Relu)
//      .layer(2, Activation::Sigmoid)
//      .outputs(5, Activation::SoftMax);
//
// or:
//
// NetBuilder::new()
//      .layer(5)
//      .relu()
//      .layer(2)
//      .sigmoid()
//      .outputs(5)
//      .softmax()
