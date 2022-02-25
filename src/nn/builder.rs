use super::layer::Layer;
use super::Network;

pub struct NetBuilder {
    num_inputs: usize,
    layers: Vec<Layer>,
}

impl NetBuilder {
    pub fn new() -> Self {
        NetBuilder {
            num_inputs: 0,
            layers: vec![],
        }
    }

    pub fn inputs(mut self, num_inputs: usize) -> Self {
        self.num_inputs = num_inputs;
        self
    }

    pub fn layer(mut self, nodes: usize, activation: Option<fn(f32) -> f32>) -> Self {
        let last_size = if let Some(layer) = self.layers.last() {
            layer.len()
        } else {
            self.num_inputs
        };
        self.layers.push(Layer::new(nodes, last_size, activation));
        self
    }

    pub fn outputs(mut self, nodes: usize, activation: Option<fn(f32) -> f32>) -> Network {
        let new = self.layer(nodes, activation);
        Network::new(new.num_inputs, new.layers)
    }
}
