use super::layer::Layer;
use super::Network;
use super::activations::Activation;

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

    pub fn layer(mut self, nodes: usize, activation: Option<Activation>) -> Self {
        let last_size = if let Some(layer) = self.layers.last() {
            layer.len()
        } else {
            self.num_inputs
        };
        self.layers.push(Layer::new(nodes, last_size, activation));
        self
    }

    pub fn init(self) -> Network {
        let mut net = Network::new(self.num_inputs, self.layers);
        net.init();
        net
    }
}
