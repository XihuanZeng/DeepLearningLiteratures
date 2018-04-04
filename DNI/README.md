# Decoupled Neural Interfaces using Synthetic Gradients
[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

The original paper was published on Aug 2016.
See [DeepMind Blog](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/) for a introduction on DNI

Forwards, update, and backwards locking constrain us to running and updating neural networks in a sequential, sync manner.
The main idea is to decouple the forward and backward propagation to achieve async update for NNs like MLP and RNN.

Core part, quote from paper

> This is achieved by using a model to approximate error gradients, rather 
> than by computing error gradients explicitly with backpropagation. 

In this paper, we remove the reliance on backpropagation to get error gradients, and instead learn a parametric model which predicts what the gradients will be based upon only local information. The synthetic gradient model takes in the activations from a module and produces what it predicts will be the error gradients - the gradient of the loss of the network with respect to the activations.

For a graphical illustration on MLP
* Use the synthetic gradients (blue) to update Layer 1 before the rest of the network has even been executed.
![N|Solid](https://github.com/XihuanZeng/DeepLearningLiteratures/blob/master/DNI/imgs/3-4.width-1500_m3lzisb.png?raw=true)

* The synthetic gradient model itself is trained to regress target gradients
![N|Solid](https://github.com/XihuanZeng/DeepLearningLiteratures/blob/master/DNI/imgs/3-5.width-1500_RuHwKed.png?raw=true)

### Paper Review
Section1: introduction
Section2: high-level communication protocol that is used to allow asynchronously learning agents to communicate. This is further break down to 
* RNN: Note that unlike CNN, RNN can be infinitely long. Ideally we want unroll every cell but our memory cannot afford this. One way is to use truncated BPTT by unrolling limit number of cells. We can incorporate a synthetic gradient model into the core so that at every time step, the RNN core produces not only the output but also the synthetic gradients. See Figure2(a) in the paper.
* CNN: We can put a synthectic model in every layer

Section3: Experiment on real dataset with different NN models
Supplimentary Details:
* section D: Implementation details on MLP and RNN


### Code Review
For the remaining part of this repo, I will put some code that I found online that use TensorFlow to implement this network architecture.



