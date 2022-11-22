title: Feed Forward Networks 
author: Heather Ann Dye
date: 11/21/2022
category: data science
tags: Pytorch

# Feed Forward Networks

Ever wondered out deep learning and neural networks work? The descriptions always contain the phrase "back propagation".  But what exactly is back propagation?  In this blog post, we'll start with a *simulated* neural network and see how it is applied to a test case. Then, we delve into how to construct and train a neural network using a hand coded model. Finally, we construct a similar network in Pytorch. 

To put together this article, I used the following references:

* Data Science from Scratch by Joel Grus - ISBN 978-1-4919 -0142-7: offers python code for many data science first principles

* [Deep Learning](deeplearningbook.org) - Goodfellow, Bengio, Courville

* [Pytorch documentation](https://pytorch.org/)

We're going to model a XOR gate. An XOR gate maps $[0,0]$ and $[1,1]$ to $0$, while $[0,1]$ and $[1,0]$ are mapped to $1$.  This gate is non-linear so we can't model our data set using linear regression (for example). Luckily, *Data Science from Scratch* has coded an XOR gate. 

##### The simulated network

We  begin with a simulated network.
$$\begin{bmatrix} \begin{bmatrix} 20 & 20 &-30 \\ 20 & 20& -10 \end{bmatrix} \\
\begin{bmatrix}-60  & 60 & -30 \end{bmatrix}\end{bmatrix}.$$

Our network consists of four layers. The hidden layer is described by the matrix
$$W =\begin{bmatrix} 20 & 20 &-30 \\ 20 & 20& -10  \end{bmatrix}.$$ 
The output layer is 
$$V = \begin{bmatrix}-60  & 60 & -30  \end{bmatrix}.$$
We use the sigmoid function for the activatation function layers. This function has the form
$$\sigma (x) = \frac{1}{1 + e^{-x}}.$$ This function has interesting properties.

* As $x$ goes to positive infinity, $\sigma (x)$ approaches $1$.
* As $x$ goes to negative infinty, $\sigma (x)$ approaches $0$. 
* $\sigma'(x) = \sigma (x) (1-\sigma (x))$.

Now, we program in our two functions to demonstrate the action of a single neuron (using code from *Data Science from Scratch*). This brief computation is the action of a single neuron. The function neuron_output takes the dot product of the two inputs and then applies the sigmoid function. 

We can simplistically describe our network as follows
$$ NN(x)= \sigma \left(  V \sigma \left( W x  \right)  \right).$$

Let's test out some values in our *simulated* network. (Bonus question: how much can we change our simulated network and still have a successful model? ) We'll start with the sigmoid function and a single neuron output.


```python
import math
import numpy as np

def sigmoid(x: float):
    return 1.0/(1+math.exp(-x))

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))

# demo neuron_output
sample_input = [0,1,1]
sample_weight = [20,20, -10]
print(neuron_output(sample_weight, sample_input))

```

    0.9999546021312976
    

Next, we use the code from *Data Science* for a feed forward network. This function will run our input through all layers of our neural network.  We compute our simulated network first and see that it models an XOR gate. 


```python
def feed_forward(neural_network, input_vector):
    outputs =[]
    for layer in neural_network: 
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs
```


```python

xor_network = [[[20,20,-30],[20,20,-10]],[[-60,60,-30]]]
for x in [0,1]:
    for y in [0,1]:
        print(feed_forward(xor_network,[x,y]))

```

    [[9.357622968839299e-14, 4.5397868702434395e-05], [9.38314668300676e-14]]
    [[4.5397868702434395e-05, 0.9999546021312976], [0.9999999999999059]]
    [[4.5397868702434395e-05, 0.9999546021312976], [0.9999999999999059]]
    [[0.9999546021312976, 0.9999999999999065], [9.383146683006828e-14]]
    

For each input, we obtained two vectors: a $1 \times 2$ vector and a $1 \times 1$ vector. The first vector is the hidden output. The $1 \times 1$ vector is our final output, so we observe all inputs are mapped correctly.  

Now, we take a look at the back propagation code from *Data Science*. This code explains how a neural network is trained. This is a bit of a misnomer, since what is really happening is minimizing a cost function. The trick to minimizing the cost function just uses things that we learned in Calculus.


```python
def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas = [output*(1-output)*(output-target)
        for output, target in zip(outputs, targets)]

    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j]-=output_deltas[i]*hidden_output
    #back-propagate errors 
    hidden_deltas = [hidden_output*(1-hidden_output)* np.dot(output_deltas,[n[i] for n in output_layer])
            for i, hidden_output in enumerate(hidden_outputs)]
    # adjust weights for hidden layer
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector+[1]):
            hidden_neuron[j]-=hidden_deltas[i]* input
```

Let's examine this code. I'm going to rewrite the neural network and construct a simplified cost function. This function depends the variables $W$ (the hidden layer) and $V$ (the output layer). 

 Use $y$ to represent our targets and $x$ to represent our input. (I'm simplifying things considerably since $V$ and $W$ are matrices.) We'll denote our cost or loss function as $C(W,V)$,
$$C(W,V) = \sum (y - NN(x))^2.$$
Some choice of $W$ and $V$ minimizes this function.

In the code snippet,  $NN(x)$ is denoted as output. We also introduce the following to denote the output:
$$\sigma_O = \sigma( V \sigma(W \: x))$$
and to denote the hidden output:
$$\sigma_h = \sigma(W \: x).$$
These sub-formulas have derivatives that resemble the sigmoid function, so $\sigma_i' = \sigma_i (1- \sigma_i)$.
Plus, they appear in the back propagation code snippet. 

##### Using some calculus

In Calculus, we learned that minimum occur when both partial derivatives are zero and that the gradient always points in the direction of steepest ascent. In back propagation, the partial derivatives of our cost function are computed. Then, we use these derivatives to push the values of $V$ and $W$ towards the location of minimum. (For various reasons, we don't have to worry about there being a variety of local minimums or saddle points.)
Now, for the output layer:
$$ \frac{\partial C}{\partial V} = 2 \sum (y - NN(X)) \: \sigma_O (1-\sigma_O) \: \sigma_h.$$

In the code,
$$output \_ deltas =output(1-output)(output-target).$$
This snippet corresponds to $(y - NN(X)) \sigma_O (1-\sigma_O)$. 

Notice that in the code the adjustment to $V$  is 

 $$output\_ neuron[j]-=output\_ deltas[i] * hidden\_ output. $$

The direction that we nudge $V$ towards corresponds to $(y - NN(X)) \sigma_O (1-\sigma_O) \sigma_h.$



Now, we consider the hidden layer.
$$\frac{\partial C}{\partial W} = 2 \sum ( y - NN(x)) ( \sigma_O)(1-\sigma_O) V \: \sigma_h (1- \sigma_h) x.$$

The adjustment for the hidden layer is: $hidden\_ neuron[j]-=hidden\_ deltas[i] (input)$ where the

 $hidden \_ deltas = hidden \_ output (1-hidden \_ output) np.dot(output\_ deltas,[n[i] \text{for n in }output \_layer]).$

 A careful examination will show the match up between the partial derivative and the adjustment to $W$. 


## Time to train our first neural network!

Now, we can use this code to set up an untrained network.
The network is initialized with random values. We observe the untrained network - the outputs don't distinguish the inputs. Then, iterate our back propagation function 1000 times over our input and targets. 


```python
targets = [[0],[0],[1],[1]]
inputs =[[0,0],[1,1],[0,1],[1,0]]
import random
random.seed(0)
input_size=2
num_hidden=2
output_size=1
hidden_layer=[[random.random() for _ in range(input_size +1)]
    for _ in range(num_hidden)]
output_layer=[[random.random() for _ in range(num_hidden+1) ]
 for _ in range(output_size)]
network=[hidden_layer, output_layer]
```
##### The untrained model

```python
for i in inputs:
    print(f"Input {i}, Output{feed_forward(network, i)[-1]}")
```

    Input [0, 0], Output[0.7561455683298192]
    Input [1, 1], Output[0.8022772919568507]
    Input [0, 1], Output[0.7845871150514832]
    Input [1, 0], Output[0.7838243382364781]
    
##### The training loop

```python
for _ in range(1000):
    for input_vector, target_vector in zip(inputs,targets):
        backpropagate(network, input_vector, target_vector)
```

Check the outputs and observe significant progress toward our target values! 


```python
for i in inputs:
    print(f"Input {i}, Output{feed_forward(network, i)[-1]}")
```

    Input [0, 0], Output[0.052840994129478264]
    Input [1, 1], Output[0.053772250071737956]
    Input [0, 1], Output[0.9513248576280653]
    Input [1, 0], Output[0.9512586782732266]
    

## Setting up the same Neural Net in Pytorch!

Now, we shift to Pytorch and build the same neural network. We start by loading packages, defining our device, and setting up the neural network. This should all look very familiar based on our earlier work.

Next, set up the feed_forward function. The bias is True, which will be the same as the  
$+[1]$ in the earlier part of our example. 


```python
import torch
from torch import nn
from torchvision import transforms
device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Sigmoid()
        self.layer3 = nn.Linear(2,1)
        self.layer4 = nn.Sigmoid()

    def forward(self,x):
        x=self.flatten(x)
        x = self.layer1(x)
        x= self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x

model = NeuralNetwork().to(device)

```

    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (layer1): Linear(in_features=2, out_features=2, bias=True)
      (layer2): Sigmoid()
      (layer3): Linear(in_features=2, out_features=1, bias=True)
      (layer4): Sigmoid()
    )
    

Next,  set up our input data and functions. Since my "data set" is so simple, it can be coded as torch tensors.

We also take a look at the predictions of the *untrained* neural network. 


```python
mydata=[]
for x in range(2):
    for y in range(2):
        local_tuple =torch.tensor([[float(x),float(y)]])
        class_val = torch.tensor([[float(x ^ y)]])
        mydata.append((local_tuple, class_val))

for batch, (X, y) in enumerate(mydata):
        # Compute prediction and loss
        pred = model(X)
        print(pred)

```

    tensor([[0.4165]], grad_fn=<SigmoidBackward0>)
    tensor([[0.4320]], grad_fn=<SigmoidBackward0>)
    tensor([[0.4042]], grad_fn=<SigmoidBackward0>)
    tensor([[0.4187]], grad_fn=<SigmoidBackward0>)
    

Next, select a loss function (MSE) and set up the optimization. Then set up a training loop. 


```python
loss_fn = nn.MSELoss()
optimize=torch.optim.SGD(model.parameters(), lr=1)

def train_loop(data, model, loss_fn, optimizer):
    size = len(data)
    for batch, (X, y) in enumerate(data):
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

       

```

Let's take one last look at the model structure before training the model.


```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

```

    Model structure: NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (layer1): Linear(in_features=2, out_features=2, bias=True)
      (layer2): Sigmoid()
      (layer3): Linear(in_features=2, out_features=1, bias=True)
      (layer4): Sigmoid()
    )
    
    
    Layer: layer1.weight | Size: torch.Size([2, 2]) | Values : tensor([[-0.4847,  0.4606],
            [-0.4183,  0.5241]], grad_fn=<SliceBackward0>) 
    
    Layer: layer1.bias | Size: torch.Size([2]) | Values : tensor([-0.1546, -0.3523], grad_fn=<SliceBackward0>) 
    
    Layer: layer3.weight | Size: torch.Size([1, 2]) | Values : tensor([[0.1152, 0.3882]], grad_fn=<SliceBackward0>) 
    
    Layer: layer3.bias | Size: torch.Size([1]) | Values : tensor([-0.5507], grad_fn=<SliceBackward0>) 
    
    
##### Model Training

```python

epochs = 1000
for t in range(epochs):
    train_loop(mydata, model, loss_fn, optimize)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



```

    Model structure: NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (layer1): Linear(in_features=2, out_features=2, bias=True)
      (layer2): Sigmoid()
      (layer3): Linear(in_features=2, out_features=1, bias=True)
      (layer4): Sigmoid()
    )
    
    
    Layer: layer1.weight | Size: torch.Size([2, 2]) | Values : tensor([[-5.1637,  5.4151],
            [-6.3016,  6.2315]], grad_fn=<SliceBackward0>) 
    
    Layer: layer1.bias | Size: torch.Size([2]) | Values : tensor([ 2.5396, -3.4433], grad_fn=<SliceBackward0>) 
    
    Layer: layer3.weight | Size: torch.Size([1, 2]) | Values : tensor([[-8.1168,  8.3146]], grad_fn=<SliceBackward0>) 
    
    Layer: layer3.bias | Size: torch.Size([1]) | Values : tensor([3.8295], grad_fn=<SliceBackward0>) 
    
    

The neural network moved substantially towards the desired outcome!


```python
for batch, (X, y) in enumerate(mydata):
        # Compute prediction 
        pred = model(X)
        print(pred)
```

    tensor([[0.0312]], grad_fn=<SigmoidBackward0>)
    tensor([[0.9720]], grad_fn=<SigmoidBackward0>)
    tensor([[0.9638]], grad_fn=<SigmoidBackward0>)
    tensor([[0.0272]], grad_fn=<SigmoidBackward0>)
    

Train the model some more and observe the predictions.


```python
epochs = 500
for t in range(epochs):
    train_loop(mydata, model, loss_fn, optimize)
for batch, (X, y) in enumerate(mydata):
        # Compute prediction and loss
        pred = model(X)
        print(pred)

```

    tensor([[0.0236]], grad_fn=<SigmoidBackward0>)
    tensor([[0.9785]], grad_fn=<SigmoidBackward0>)
    tensor([[0.9728]], grad_fn=<SigmoidBackward0>)
    tensor([[0.0207]], grad_fn=<SigmoidBackward0>)
    

We've successfully built a neural network for an XOR gate in Pytorch!
