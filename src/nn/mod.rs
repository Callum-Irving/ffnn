pub mod activations;
mod builder;
mod layer;
mod network;

pub use self::builder::NetBuilder;
pub use self::network::Network;

type Float = f32;

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
