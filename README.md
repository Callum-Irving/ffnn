# FFNN - a very simple feedforward neural network library

Not useful yet, still a WIP.

## TODO

- [ ] Add documentation

- [ ] Backward propagation
- [ ] Batched backprop
- [ ] Adam optimizer
- [ ] Softmax activation
- [ ] Dropout
- [ ] Genetic algorithm utils?
- [ ] Multiple cost functions
    - Binary cross entropy
    - Categorical cross entropy
    - Mean squared error
- [ ] Maybe multiple optimization methods (not just gradient descent)
- [ ] Thread pool to increase performance
- [ ] Use GPU

### Create some examples

- MNIST digits
- XOR function
- Sine wave

### Done

- [x] Forward propagation
- [x] Add bias nodes
- [x] Move initialization to the builder
- [x] Multiple activation functions

## References

- [nalgebra docs](https://nalgebra.org/docs/user_guide/vectors_and_matrices)
- [The Math behind Neural Networks - Backpropagation](https://www.jasonosajima.com/backprop)
- [Why Initialize a Neural Network with Random Weights?](https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/)
- [A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
