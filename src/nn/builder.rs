use super::layer::Layer;

pub struct NetBuilder {
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
