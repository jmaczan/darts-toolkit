# A primer on Differentiable Architecture Search

## In a nutshell

Differentiable Architecture Search is a thing that comes up with an architecture of a neural network for given training data. Unlike the traditional approach in which we rely on humans to design an architecture by hand, here we use gradient descent to automate the architecture search. This is the same mathematical optimization as for training neural networks. Once we find the good enough architecture, we can use it to train the network.

## Important concepts and structure of the algorithm

The architecture of a neural network that DARTS finds is called a `cell`. It's a repeatable building block of an architecture. Repeatable because we can stack multiple cells on top of each other to build a deeper network. A cell consists of `nodes`. A node stores features tensor. First node stores input features. Intermediate nodes store intermediate activations. Last node stores output of a cell. Nodes are connected with `edges`. An edge contains three things. A first one is a collection of allowed operations stored as a single tensor in a `mixed operation`. We call these operations `candidate operations`. The second element of an edge are `architecture parameters` `alpha`, which are real positive values. The value of architecture parameter tells how much the particular operation contributes to the network output. To me it's like an importance of an operation. In each edge, each candidate operation have exactly one corresponding architecture parameter. The last part of the edge are `network parameters`, which are weights and biases for each candidate operation that support trainable parameters. Some operations, like convolutions, have trainable parameters, and others, like max pooling, don't. To preempt any confusion, let's emphasize that network parameters are not architecture parameters. Network parameters are trainable numbers for each operation. Every neural network have them, both those constructed by a human expert and those built automatically by DARTS algorithm. However, the architecture parameters are exclusive to DARTS. They represent the importance each of candidate operation in each edge.

## How to search

Both architecture and network parameters are being modified during architecture search. We train the network parameters, like weights and biases, using architecture parameters. Then, we train the architecture parameters using network parameters. We alternate between these two optimizations until we get satisfying results or run out of resources (time, budget, etc.). Sensibly, this kind of training is called `bi-level optimization`.

At the beginning of the architecture search, all nodes are connected with edges to all preceeding them nodes. Architecture parameters (alphas) in edges are initialized with small random values. Likewise, the candidate operations that have trainable parameters are initialized with random values.

In the end, in order to form the final architecture, at every edge we pick the candidate operation that has the highest architecture parameter (alpha).

Once architecture search is done, each edge is exactly a single operation (like 3x3 convolution, 5x5 max pooling etc.).
