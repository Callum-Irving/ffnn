//! # ffnn
//!
//! **ffnn** is meant to be a very simple feedforward neural network library. The
//! goal is to make creating and training feedforward neural networks as easy
//! as possible.
//!
//! # Getting Started
//!
//! Make a neural network using the [NetBuilder] struct.
//!
//! ```
//! let net = NetBuilder::new(3).layer(5, RELU).init();
//! ```
//!
//! Making predictions on a set of inputs is as simple as:
//!
//! ```
//! let result = net.predict(dvector![0.0, 0.0, 0.0]);
//! ```

#![warn(missing_docs)]

mod nn;

pub use nn::activations;
pub use nn::losses;
pub use nn::NetBuilder;
pub use nn::Network;
