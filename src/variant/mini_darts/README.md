# Educational DARTS implementation
# A primer on Differentiable Architecture Search

We build a thing that to finds a good architecture of a neural network for given training data. Instead of relying on a human to come up with an architecture, it automates this process. To find the architecture, it uses gradient descent. This is the same mathematical optimization as in training neural networks. Once we find the right architecture, we train the network. 

The method we use is called Differentiable Architecture Search. It's differentiable, because it's possible to 

An architecture of a neural network that DARTS `cell`. It's a repeatable block of a network architecture. We can stack multiple cells on top of each other to construct a deeper network. A cell consists of nodes. A node stores cell parameters (current values of weights and biases of the network). A first node stores input features. Intermediate nodes store hidden parameters. A last node stores output of a network. Nodes are connected with edges. At the beginning of architecture search, all nodes are connected (with edges) to all preceeding them nodes. Edges are   Once architecture search is done, each edge is exactly a single operation (like 3x3 convolution, 3x3 max pooling etc.). But during the search, each edge holds multiple operations at once. It's called Mixed Operation. 

