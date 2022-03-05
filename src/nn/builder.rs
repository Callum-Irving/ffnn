use super::activations::Activation;
use super::layer::Layer;
use super::Network;

/// A struct for using the builder pattern to define a network.
///
/// TODO: Add basic example
pub struct NetBuilder {
    num_inputs: usize,
    layers: Vec<Layer>,
}

impl NetBuilder {
    /// Initialize a new network with `num_inputs` inputs.
    pub fn new(num_inputs: usize) -> Self {
        NetBuilder {
            num_inputs,
            layers: vec![],
        }
    }

    /// Add a new layer to the network.
    pub fn layer(mut self, nodes: usize, activation: Activation) -> Self {
        let last_size = if let Some(layer) = self.layers.last() {
            layer.len()
        } else {
            self.num_inputs
        };
        self.layers.push(Layer::new(nodes, last_size, activation));
        self
    }

    /// Randomly initialize the network.
    pub fn init(self) -> Network {
        let mut net = Network::new(self.num_inputs, self.layers);
        net.init();
        net
    }
}
