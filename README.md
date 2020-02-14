## Project Description

An MLP Neural Network Implementation using only NumPy. 

## Usage

```python

# 1. Import from Neural_Nework.py
from Neural_Nework import *

# 2. Define Architecture (python list)
architecture = [784,100, 10]

# 3. Instatiate Neural Network and Evalutation Metrics Objects
nn = MLP_Neural_Network(architecture) # archtiecture
em = Evaluation_Metrics()

# 4. Forward pass
nn.forward(x)   # x: (np.array) input array, Shape: [Batch Size, Features=784]

# 5. Evaluate prediction
loss = em.Cross_Entropy_Loss(pred, y)
acc = em.Multi_Class_Classificiton_Accuracy(pred, y)

# 6. Backward pass
nn.backward(y)

# 7. Gradient Descent Step
nn.update_weights()

``` 

## Backpropagation Derviations

The derviations below are used to obtain a vectorized result. These are also available in derviations.pdf


<table style="width:70%">
  <tr>
   <img src= "https://github.com/mgamal96/Neural-Network-Numpy-Implementation/blob/master/derivation%20imgs/bp1.png?raw=true">
  </tr>
  
  <tc>
   <img src= "https://github.com/mgamal96/Neural-Network-Numpy-Implementation/blob/master/derivation%20imgs/bp2.png?raw=true">
  </tc>
  
   <tc>
   <img src= "https://github.com/mgamal96/Neural-Network-Numpy-Implementation/blob/master/derivation%20imgs/bp3.png?raw=true">
  </tc>
  
  <tc>
  <img src= "https://github.com/mgamal96/Neural-Network-Numpy-Implementation/blob/master/derivation%20imgs/bp4.png?raw=true">
  </tc>
  
</table>
