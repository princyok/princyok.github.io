---
layout: post
title:  "Catching AI with its pants down: Implement an Artificial Neuron from Scratch"
logline: "Going from equations to implementation in Python."
date:   "2020-04-04"
categories: machine-learning
permalink:
comments: true
---
{% include scripts.html %}

{% include blogseries_mantra_catching_ai.html %}

* TOC
{:toc}

## **Prologue**

This is part 4 of the blog series, *Catching AI with its pants down*. This blog series aims to explore the inner workings of neural networks and show how to build a standard feedforward neural network from scratch.

In this part we will implement all the equation that we derived from scratch in the previous parts.

{% include blogseries_index_catching_ai.html %}

## **Code Implementation: an artificial neuron**

All the codes will be in Python, using its object-oriented paradigm wherever possible (but I won’t bother with [getters and setters](https://en.wikipedia.org/wiki/Mutator_method){:target="_blank"} for the most part). We will use primarily the [NumPy library](https://numpy.org/){:target="_blank"} because its operations are very efficient for linear algebra computations involving arrays.

This implementation does not take advantage of parallel computing, so your GPU won’t make things any faster. But it takes advantage of NumPy’s superb optimization for computations with multidimensional arrays. Therefore, python loops are avoided as much as possible in the code, which is why we went through all that work to have everything as tensors.

We will also not implement any concurrent computing (so no multithreading of any sort) other than any that may have been baked into NumPy. Most deep learning libraries include concurrent and parallel computing capabilities, and also automatic differentiation capability. Moreover, none of those are really needed for a single artificial neuron. But they are absolutely priceless when training a network of neurons (a.k.a. neural network).

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

We don’t really need to access the entire data (`X` and `Y`) during its instantiation. We could have chosen to initialize `self.X` and `self.Y` later. We only just needed the shape of `X`, because we use it to get the number of features in our data which we use when we initialize our parameters. However, I chose to have both `self.X` and `self.Y` initialized at instantiation for the sake of it, so this is certainly an opportunity for some refactoring to improve the code.

### **Parameter initialization**

We first initialize our parameters, and we will do this randomly.

Next, we will implement a method for parameter initialization. It’s just going to be plain random initialization.

```python
def _initialize_parameters(self, random_seed=11):
    prng=np.random.RandomState(seed=random_seed)
    n=self.X.shape[0]
    self.w=prng.random(size=(1, n))*0.01
    self.b=np.zeros(shape=(1, 1))
```

### **Forward pass**
Forward pass can be broken into two steps: First is the linear combination of the parameters and datapoint values to get the preactivation. Next is the passing of the preactivation through an activation function to get the activation.

The equations for forward pass are:

$$
\vec{z}=\vec{w}\mathbf{X}+b
$$

$$
\vec{a}=f\left(\vec{z}\right)
$$

```python
def _forward(self):
    self.z = np.matmul(self.w, self.X_batch) + self.b
    self.a=self._logistic(self.z)
```

Notice that that I used `self.X_batch` instead of `self.X`, because we perform our calculations on batches of samples from the dataset. We will initialize `self.X_batch` during training (i.e. inside the `train` method).

#### **Activation function**

Next, we implement out activation function. We will only do logistic for this model of an artificial neuron. Check out the deep neural network code for some other activation functions.

$$
\vec{a}=f\left(\vec{z}\right)=\frac{1}{1+e^{-\vec{z}}}
$$

```python
def _logistic(self, z):
    a = 1/(1+np.exp(-z))
    return a
```
We will also implement the derivate of the activation function (we are using the logistic function). But note that we invoke this method only during backward pass, not forward pass. Presenting it here (and writing the code near that for the forward pass) is just a matter of personal taste.

$$
f'\left(\vec{z}\right)=\vec{a}\odot\left(1-\vec{a}\right)
$$

```python
def _logistic_gradient(self, a):
    dAdZ = a * (1-a)
    return dAdZ
```

### **Calculation of Cost**

Next, we should implement the method for computing the cost, but I didn’t do it for artificial neuron, but instead did it for the main thing, the deep neural network code, and the blog post for it is coming soon.

Note that you don’t actually need the cost for the training process, but instead the cost gradients. The cost is just there to tell us how the training is progressing.
This is the equation we would implement.

$$
J=-\frac{1}{m}\bullet\sum_{j}^{m}{y_i\cdot \log{(y}_i)+(1-a_i)\bullet\log({1-a}_i)}
$$

### **Backward pass**

Now we will optimize our parameters in such a way that our loss decreases. We start by first computing the cost gradient $$\frac{\partial J}{\partial\vec{w}}$$:

$$
\frac{\partial J}{\partial\vec{w}}=\frac{\partial J}{\partial\vec{z}}\frac{\partial\vec{z}}{\partial\vec{w}}=\ \frac{\partial J}{\partial\vec{z}}X^T=\frac{\partial J}{\partial\vec{a}}\odot\frac{\partial\vec{a}}{\partial\vec{z}}X^T=\frac{\partial J}{\partial\vec{a}}\odot f'(\vec{z})X^T
$$

For a logistic loss function and a logistic activation function, we have:

$$
\frac{\partial J}{\partial\vec{w}}=-\frac{1}{m}\bullet\left(\frac{\vec{y}}{\vec{a}}-\frac{1-\vec{y}}{1-\vec{a}}\right)\ \odot(\vec{a}\odot\left(1-\vec{a}\right))X^T
$$

We could directly implement the above equation, but I chose to implement it in stages, with each gradient computed at each stage. This will make it a little easier to swap in other activation functions and loss functions in the future (I don’t really have any intention to do so for the artificial neuron code, as I already did it in the deep neural network code).

So, we implement the following equations step by step:

$$
\frac{\partial\vec{a}}{\partial\vec{z}}:=f'\left(\vec{z}\right)=\vec{a}\odot\left(1-\vec{a}\right)
$$

$$
\frac{\partial J}{\partial\vec{a}}=-\frac{1}{m}\bullet\left(\frac{\vec{y}}{\vec{a}}-\frac{1-\vec{y}}{1-\vec{a}}\right)
$$

$$
\frac{\partial J}{\partial\vec{z}}=\frac{\partial J}{\partial\vec{a}}\odot\frac{\partial\vec{a}}{\partial \vec{z}}
$$

$$
\frac{\partial J}{\partial\vec{w}}=\ \frac{\partial J}{\partial\vec{z}}X^T
$$

The cost gradients for the bias is:

$$
\frac{\partial J}{\partial b}=\sum_{j=1}^{m}\left(\frac{\partial J}{\partial\vec{z}}\right)_j
$$

As we showed in part 3, we can also choose to use this equation instead:

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial\vec{z}}\ \frac{\partial\vec{z}}{\partial b}
$$

{% include indent_paragraph.html content=
"Where $ \frac{\partial \vec{z}}{\partial b} $ is an $ m $-by-$ 1 $ vector of ones (i.e. has same shape as $ \vec{z}^T $)."
%}

Both equations, implemented as `self.dJdB= np.sum(self.dJdZ, axis=1)` and `self.dJdB= np.matmul(self.dJdZ, np.ones(self.z.T.shape))`, produce the same result. We will use the former.

```python
def _backward(self):
    m = self.X_batch.shape[1]
    self.dAdZ=self._logistic_gradient(self.a)
    self.dJdA = - (1/m) *((self.Y_batch / self.a) - ((1 - self.Y_batch) / (1 - self.a)))

    self.dJdZ = self.dAdZ * self.dJdA

    self.dJdW= np.matmul(self.dJdZ, self.X_batch.T)
    self.dJdB= np.sum(self.dJdZ, axis=1)
```
#### **Update parameters via gradient descent**

Next, we update *each* parameter using gradient descent:

$$
w_{new}=w_{old}-\gamma\frac{\partial J}{\partial w_{old}}
$$

$$
b_{new}=b_{old}-\gamma\frac{\partial J}{\partial b_{old}}
$$

{% include indent_paragraph.html content=
"Where $ \gamma $ is the learning rate (a.k.a. step size). It's a hyperparameter, meaning that it is a variable you directly set and control.
<br><br>
Note that $ \frac{\partial J}{\partial w_{old}} $ is simply the $ \frac{\partial J}{\partial w} $ that we just calculated, and the same is true for $ \frac{\partial J}{\partial b_{old}} $."
%}

With this, we’ve completed one iteration of training. We repeat this as many times as we want. Eventually, we expect to end up with an artificial neuron that has learned the underlying relationship between the features and the target.

```python
def _update_parameters_via_gradient_descent (self, learning_rate):
    self.w = self.w - learning_rate * self.dJdW
    self.b = self.b - learning_rate * self.dJdB
```

### **Training**

The training process is as follows:
1.	Randomly initialize our parameters
2.	Run one iteration of training, which involves:
  *	Sample a batch from our dataset.
  *	Then run forward pass (i.e. move the data forward through the neuron).
  *	Then run backward pass to calculate our cost gradients.
  *	Then run gradient descent (which is technically part of backward pass), which uses the cost gradients to update the parameters.
3. Repeat step 2 until we reach the specified number of iterations.

Therefore we combine the code snippets accordingly:

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

We have three hyperparameters we can use to tune the training process: number of iterations, learning rate, and batch size.

### **Evaluation of trained artificial neuron**

And finally, we implement methods for evaluating the neuron, including method for computing accuracy and precision. These are very pretty straightforward.

```python
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

We bundle the two methods under on method for evaluating the model:

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

I decided to get a little cheeky and throw `ValueError` when an invalid string is passed to `metric`, a formal parameter of the method evaluate.

I also decided to print a warning message if all my activations are [NaNs](https://docs.scipy.org/doc/numpy-1.13.0/user/misc.html){:target="_blank"} (i.e. null values). From my experience, these can occur when the computations cause an arithmetic overflow or underflow.

## **All the codes**

You can find the entire code, along with the code for deep neural network (the writeup for it is coming soon) and demonstrations using it to tackle real public research datasets, in [**this GitHub repo**](https://github.com/princyok/deep_learning_without_ml_libraries){:target="_blank"}.

The version as of the end of March 2020 is repeated here for your convenience:

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

See you in the next article!
