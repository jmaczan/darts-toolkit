# A primer on Differentiable Architecture Search

## In a nutshell

We build a thing that comes up with an architecture of a neural network for given training data. Traditionally, we rely on humans to design with an architecture by hand. To find the architecture, it uses gradient descent. This is the same mathematical optimization as in training neural networks. Once we find the right architecture, we train the network using this architecture.

## Important concepts and structure of the algorithm

An architecture of a neural network that DARTS `cell`. It's a repeatable block of a network architecture. We can stack multiple cells on top of each other to construct a deeper network. A cell consists of `nodes`. A node stores features tensor. A first node stores input features. Intermediate nodes store intermediate activations. A last node stores output of a cell. Nodes are connected with `edges`. An edge contains three things. First is a `mixed operation`, which is a collection of operations that become parts of the architecture. We call them `candidate operations`. The second thing in an edge are architecture parameters `alpha`, which are real positive values. The value of `architecture parameter` tells how much the operation contributes to the network output. I think of it as an importance of an operation. Each candidate operation have exactly one corresponding architecture parameter. The last part of the edge are `network parameters`, like weights and biases for each candidate operation that have trainable parameters. Some operations have trainable parameters, like convolutions, and others don't, like max pooling. To preempt any confusion, let's remind that network parameters are not architecture parameters. Network parameters are trainable numbers for each operation. All network architectures have them, both those constructed by a human expert and those built automatically by an algorithm like this one. On the other hand, architecture parameters are exclusive to DARTS. They represent the importance each of candidate operation in each edge.

## How to search

Both architecture and network parameters are being modified during architecture search. We train network parameters, like weights and biases, using architecture parameters. Then, we train architecture parameters using network parameters. We keep alternating these two steps of optimization until we get good enough results or run out of time. Sensibly, it is called `bi-level optimization`. In the end, to form the final architecture, in every edge we pick the candidate operation that has the highest architecture parameter (alpha).

At the beginning of the architecture search, all nodes are connected with edges to all preceeding them nodes. Architecture parameters (alphas) in edges are initialized with small random values. Likewise, the candidate operations that have trainable parameters are initialized with random values.

Once architecture search is done, each edge is exactly a single operation (like 3x3 convolution, 5x5 max pooling etc.).
