---
layout: post
title:  "Catching AI with its pants down: Implement an artificial neuron from scratch."
logline: "Going from equations to code implementation in Python."
date:   "2020-04-01"
categories: machine-learning
permalink:
comments: true
---
{% include scripts.html %}

<table>
<td>
<i>We will strip the mighty, massively hyped, highly dignified AI of its cloths, and bring its innermost details down to earth!</i>
</td>
</table>

* TOC
{:toc}



### **Summary of workflow**

The workflow is as follows:

#### **Initialize parameters**

We first initialize our parameters, and we will do this randomly.

However, there are a handful of initialization schemes out there that promise to initialize our parameters with values that enable the network to be optimized faster. You can think of it as beginning a journey from a point closer to the destination, which allows you to finish faster than another person who began much farther away because they chose to randomly begin at some arbitrary point.

One such scheme for sigmoid activation functions (e.g. logistic, hyperbolic tangent, etc.), the Xavier Initialization Scheme (introduced by [Xavier Glorot in 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), is implemented in my code for deep neural networks.

#### **Run forward pass**

During forward pass, we compute our activation $$a$$ as follows (we use a logistic activation function):

$$
 \boldsymbol{z}= \boldsymbol{w}X+ \boldsymbol{b}
$$

$$
 \boldsymbol{a}=f\left( \boldsymbol{z}\right)=\frac{1}{1+e^{- \boldsymbol{z}}}
$$

Then we check our activation $$a$$ against the ground truth $$y$$ and compute the corresponding cost.

$$
J=-\frac{1}{m}\cdot\sum_{j}^{m}{y_i\cdot l o g{(y}_i)+(1-a_i)\cdot\log({1-a}_i)}
$$

But we will not directly use this cost any further in this workflow. We just need it in order to keep track of how good how network is doing.

#### **Optimize parameters**

Now we will optimize our parameters in such a way that our loss decreases. The backward pass begins here. We start by first computing the cost gradient $$\frac{\partial J}{\partial w}$$:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T=\left(-\frac{1}{m}\cdot\left(\frac{ \boldsymbol{y}}{ \boldsymbol{a}}-\frac{1- \boldsymbol{y}}{1- \boldsymbol{a}}\right)\ \odot(a\odot\left(1-a\right))\right)X^T
$$

The beautiful thing about the above equation is that everything on the right-hand side has already been computed during forward pass, so this saves us a considerable amount of computation. This benefit becomes even more apparent when doing a network of artificial neurons, and that is the power of backpropagation.

Next, we update our parameters using gradient descent:

$$
w_{new}=w_{old}-\gamma\frac{\partial J}{\partial w_{old}}
$$

$$
b_{new}=b_{old}-\gamma\frac{\partial J}{\partial b_{old}}
$$

Note that $$\frac{\partial J}{\partial w_{old}}$$ is simply the $$\frac{\partial J}{\partial w}$$ that we just calculated prior, and the same is true for $$\frac{\partial J}{\partial b_{old}}$$.

With this, we’ve completed one iteration of training. We repeat this as many times as we want. Eventually, we expect to end up with an artificial neuron that has learned the underlying relationship between the features and the target.

### **Code: Implementation: An artificial neuron**

All that remains now is to code the workflow that we just described above.

All the codes that will be done in this treatment will be in python, using its object-oriented paradigm wherever possible (but I won’t bother with getters and setters, for the most part).

Subsequent explanations assume that you’re familiar with the python library NumPy. If you’re coming from Java and are familiar with Spark Dataframe API for Java, then you may be able to relate. Also, if you’re coming from MATLAB, you should be able to follow along too. I don’t know any opensource equivalent for C#, but as for C++, if you’re familiar with xtensor and already understand the basics of python, it should be possible to follow along.

This implementation does not take advantage of parallel computing, so your GPU won’t make things any faster. But it takes advantage of NumPy’s superb optimization for tensor computations; therefore, python loops are avoided as much as possible, which is why we went through all that work to have everything as tensors, so that Numpy can do what its good at. We also do not implement any concurrent computing (so no multithreading of any sort) other than those that are baked into NumPy. Production-grade deep learning libraries typically include options for concurrent and parallel computing capabilities.

#### **Constructor**

We begin by implementing our constructor, where we initialize all our data members (also using it as an opportunity to lay them all out).

{% highlight python %}
import numpy as np

class Neuron:
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y

        self.X_batch=None
        self.Y_batch=None

        self.a=None
        self.z=None
        self.w=None
        self.b=None

        self.dAdZ=None
        self.dJdA=None
        self.dJdZ=None
        self.dJdW=None
        self.dJdB=None
{% endhighlight %}

We don't really need the object to access the data (X and Y) during its instantiation,. We only just needed the shape of X. But I chose to have both X and Y initialized at instantiation for the sake of it, so this may be an opportunity for some constructive refactoring.

#### **Activation function**

Next, we implement the activation function. We will only do the logistic function for artificial neuron. But check out the deep neural network code for some other activation functions.

```python
def _logistic(self, z):
    a = 1/(1+np.exp(-z))
    return a
```

We will also implement the derivate of the activation function.

```python
def _logistic_gradient(self, a):
    dAdZ = a * (1-a)
    return dAdZ
```

#### **Parameter initialization**

Next, we will implement a method for parameter initialization. It's just going to be plain random initialization.

```python
def _initialize_parameters(self, random_seed=11):
    prng=np.random.RandomState(seed=random_seed)
    n=self.X.shape[0]
    self.w=prng.random(size=(1, n))*0.01
    self.b=np.zeros(shape=(1, 1))
```

#### **Forward pass**
Next, we will implement a method for the forward pass. Note that we use `X_batch` instead of `X`, because we perform our calculations on batches of samples from the dataset.

```python
def _forward(self):
    self.z = np.matmul(self.w, self.X_batch) + self.b
    self.a=self._logistic(self.z)
```

### **Calculation of Cost**
Then we should implement the method for computing the cost, but I didn’t do it for artificial neuron, but instead did it for the deep neural network code. Note that you don’t actually need the cost for the training process, but instead the cost gradients.

### **Backward pass**

Then we will implement method for backward pass.

```python
def _backward(self):
    m = self.X_batch.shape[1]
    self.dAdZ=self._logistic_gradient(self.a)
    self.dJdA = - (1/m) *((self.Y_batch / self.a) - ((1 - self.Y_batch) / (1 - self.a)))

    self.dJdZ = self.dAdZ * self.dJdA

    self.dJdW= np.matmul(self.dJdZ, self.X_batch.T)
    self.dJdB= np.sum(self.dJdZ, axis=1)
```
Implement the updating of parameters:

```python
def _update_parameters_via_gradient_descent (self, learning_rate):
    self.w = self.w - learning_rate * self.dJdW
    self.b = self.b - learning_rate * self.dJdB
```

### **Training**
Then we will put everything together in the method for training.

```python
def train(self,num_iterations, learning_rate, batch_size, random_seed=11):
    print("Training begins...")
    self._initialize_parameters(random_seed=random_seed)
    prng=np.random.RandomState(seed=random_seed)
    for i in range(0, num_iterations):
        random_indices = prng.choice(self.Y.shape[1], (batch_size,), replace=False)
        self.Y_batch = self.Y[:,random_indices]
        self.X_batch = self.X[:,random_indices]

        self._forward()
        self._backward()

        self._update_parameters_via_gradient_descent(learning_rate=learning_rate)

    print("Training Complete!")
```
### **Evaluation of trained artificial neuron**

And finally, we implement methods for evaluating the network, including method for computing accuracy.
```python
def evaluate(self, X, Y, metric="accuracy"):

    _available_perfomance_metrics=["accuracy","precision"]

    metric=metric.lower()

    if not any(m == metric.lower() for m in _available_perfomance_metrics):
        raise ValueError

    self.X_batch = X
    self.Y_batch = Y

    self._forward()

    if metric=="accuracy":
        score=self._compute_accuracy()
    if metric =="precision":
        score=self._compute_precision()

    return score
```

### **All the codes**
Here is the entire code:
```python
import numpy as np
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

class Neuron:
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y

        self.X_batch=None
        self.Y_batch=None

        self.a=None
        self.z=None
        self.w=None
        self.b=None

        self.dAdZ=None
        self.dJdA=None
        self.dJdZ=None
        self.dJdW=None
        self.dJdB=None

    def _logistic(self, z):
        a = 1/(1+np.exp(-z))
        return a

    def _logistic_gradient(self, a):
        dAdZ = a * (1-a)
        return dAdZ

    def _forward(self):
        self.z = np.matmul(self.w, self.X_batch) + self.b
        self.a=self._logistic(self.z)

    def _backward(self):
        m = self.X_batch.shape[1]
        self.dAdZ=self._logistic_gradient(self.a)
        self.dJdA = -(1/m) * ((self.Y_batch / self.a) - ((1 - self.Y_batch) / (1 - self.a)))

        self.dJdZ = self.dAdZ * self.dJdA

        self.dJdW= np.matmul(self.dJdZ, self.X_batch.T)
        self.dJdB= np.sum(self.dJdZ, axis=1)

    def _update_parameters_via_gradient_descent (self, learning_rate):
        self.w = self.w - learning_rate * self.dJdW
        self.b = self.b - learning_rate * self.dJdB

    def _initialize_parameters(self, random_seed=11):
        prng=np.random.RandomState(seed=random_seed)
        n=self.X.shape[0]
        self.w=prng.random(size=(1, n))*0.01
        self.b=np.zeros(shape=(1, 1))

    def _compute_accuracy(self):

        if np.isnan(self.a).all():
            print("Caution: All the activations are null values.")
            return None

        Y_pred=np.where(self.a>0.5, 1, 0)
        Y_true=self.Y_batch

        accuracy=np.average(np.where(Y_true==Y_pred, 1, 0))

        return accuracy

    def _compute_precision(self):

        if np.isnan(self.a).all():
            print("Caution: All the activations are null values.")
            return None

        Y_true=self.Y_batch
        Y_pred=np.where(self.a>0.5, 1, 0)

        pred_positives_mask = (Y_pred==1)
        precision=np.average(np.where(Y_pred[pred_positives_mask]==Y_true[pred_positives_mask]))        

        return precision

    def train(self,num_iterations, learning_rate, batch_size, random_seed=11):
        print("Training begins...")
        self._initialize_parameters(random_seed=random_seed)
        prng=np.random.RandomState(seed=random_seed)
        for i in range(0, num_iterations):
            random_indices = prng.choice(self.Y.shape[1], (batch_size,), replace=False)
            self.Y_batch = self.Y[:,random_indices]
            self.X_batch = self.X[:,random_indices]

            self._forward()
            self._backward()

            self._update_parameters_via_gradient_descent(learning_rate=learning_rate)

        print("Training Complete!")

    def evaluate(self, X, Y, metric="accuracy"):

        _available_perfomance_metrics=["accuracy","precision"]

        metric=metric.lower()

        if not any(m == metric.lower() for m in _available_perfomance_metrics):
            raise ValueError

        self.X_batch = X
        self.Y_batch = Y

        self._forward()

        if metric=="accuracy":
            score=self._compute_accuracy()
        if metric =="precision":
            score=self._compute_precision()

        return score
```
You can find it, along with the code for part 3 (deep neural network), in [this GitHub repo](https://github.com/princyok/deep_learning_without_ml_libraries).

## **Part 2: Building a Feedfoward Deep Neural Network from Scratch**

To be continued: It should be both shorter and easier to write than part 1, since I already set up in part 1 much of what I need for part 2. However, the code for part 2 is available on my GitHub, including multiple demonstrations of using it.
