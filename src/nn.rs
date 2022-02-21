use nalgebra::DMatrix;

struct Layer {
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

struct Network {
    inputs: Layer,
    hidden: Vec<Layer>,
    outputs: Layer,
}

impl Network {
    pub fn new(inputs: usize, hidden: Vec<Layer>, outputs: Layer) -> Self {
        Network {
            inputs: Layer::new(inputs, None),
            hidden,
            outputs,
        }
    }

    pub fn predict(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(self.inputs.len(), inputs.len());
        todo!();
    }

    pub fn train(&mut self, dataset: DMatrix<f32>) {
        // Number of columns in dataset should match length of inputs
        assert_eq!(dataset.ncols(), self.inputs.len());

        todo!();
    }
}

struct NetBuilder {
    layers: Vec<Layer>,
}

impl NetBuilder {
    pub fn new() -> Self {
        NetBuilder { layers: vec![] }
    }

    pub fn layer(&mut self, nodes: usize, activation: Option<fn(f32) -> f32>) {
        self.layers.push(Layer::new(nodes, activation));
    }
}

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
