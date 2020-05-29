---
layout: post
title:  "Catching AI with its pants down: Implement a Neural Network from Scratch"
logline: "Going from equations to implementation in Python."
date:   "2020-05-13"
categories: machine-learning
permalink:
comments: true
---
{% include scripts.html %}

{% include blogseries_mantra_catching_ai.html %}

* TOC
{:toc}


## **Prologue**

This is part 7 of this blog series, *Catching AI with its pants down*. This blog series aims to explore the inner workings of neural networks and show how to build a standard feedforward neural network from scratch. In this part, we will implement all the equations that we derived in parts 5 and 6.

{% include blogseries_index_catching_ai.html %}

### **Heads up**

All the codes will be in python, using its object-oriented paradigm wherever possible (but I won’t bother with getters and setters, for the most part). We will use primarily the NumPy library because its operations are very efficient for linear algebra computations involving arrays.
 
This implementation does not take advantage of parallel computing, so your GPU won’t make things any faster. But it takes advantage of NumPy’s superb optimization for computations with multidimensional arrays. Therefore, python loops are avoided as much as possible in the code, which is why we went through all that work to have everything as tensors. 

We will also not implement any concurrent computing (so no multithreading of any sort) other than any that may have been baked into NumPy. Most deep learning libraries include concurrent and parallel computing capabilities, and also automatic differentiation capability.

## **Overview of the code implementation**

There are six classes, namely `Layer`, `_InputLayer`, `_ActivationFunction`, `TrainingArchiver`, `ParameterInitializer`, and `Network`.
 
The `Layer` class controls all the data and operations for each layer of the network. The `_InputLayer` class is a private class for the input layer (the input data). It inherits from the `Layer` class.

The `_ActivationFunction` class handles operations involving the activation functions. It is where all the activation functions are implemented. The `ParameterInitializer` class handles the initialization of the parameters of the neural network.

The `TrainingArchiver` class takes care of all caching and archiving of data generated during the training process, as well as any periodic printing of relevant data (e.g. training/validation loss) to console during training. All the data cached by a training archiver can be retrieved after training.

The `Network` class is the assembly point for the entire program. It represents the neural network. All the other classes feed into the `Network` class. It’s were operations that apply to the entire network, like the training logic, are implemented.

### **Layers**

A `Layer` instance represents a layer. It holds all the data for a layer, which includes the parameters, gradients, activations and preactivations associated with that layer. It also knows the number of units in the layer, the network it belongs to, its serial position in that network (whether it is the 0<sup>th</sup> or 1<sup>st</sup> or $$l$$<sup>th</sup> layer), and it has an attribute that points to the layer preceding it in that network. 

{% highlight python %}
class Layer:
    def __init__(self, activation_name, num_units):
        
        if activation_name == None:
            self.activation_type=None            
        else:
            self.activation_type=_ActivationFunction(name=activation_name)
            
        self.num_units=num_units
        self.W=None
        self.B=None
        self.A=None
        self.Z=None
        self.gradients=dict(zip(["dAdZ", "dJdA", "dJdZ", "dJdW", "dJdB"],[None]*5))
        self.parent_network=None
        self.position_in_network=None
        self.preceding_layer=None
{% endhighlight %}

The Layer can perform both forward and back propagation on itself.

The equation for forward propagation implemented in the Layer class is:

$$
\mathbf{Z}^{(l)}=\mathbf{W}^{(l)}\mathbf{A}^{(l-1)}+\mathbf{B}^{(l)}
$$

{% highlight python %}
def _compute_linear_preactivation(self): # class-private.
	A_prior = self.preceding_layer.A

	self.Z = np.matmul(self.W, A_prior) + self.B
				
	return None
	
def _layer_forward_prop(self): # module-private.
	self._compute_linear_preactivation()
	
	self.A=self.activation_type._forward_pass(self.Z)
	
	return None

{% endhighlight %}

The equations for computing cost gradients (which you can think of as “layer-level” back propagation) for a hidden layer that are implemented inside the Layer class are:

$$
\frac{\partial J}{\partial\mathbf{Z}^{\left(l\right)}}=\frac{\partial J}{\partial\mathbf{A}^{\left(l\right)}}\odot f^\prime\left(\mathbf{Z}^{\left(l\right)}\right)
$$

$$
\frac{\partial J}{\partial\mathbf{B}^{(l)}}=\sum_{j=1}^{m}\left(\frac{\partial J}{\partial\mathbf{Z}^{(l)}}\right)_j
$$

$$
\frac{\partial J}{\partial\mathbf{W}^{(l)}}=\frac{\partial J}{\partial\mathbf{Z}^{(l)}}\mathbf{A}^{(l-1)T}
$$

$$
\frac{\partial J}{\partial\mathbf{A}^{\left(l-1\right)}}=\mathbf{W}^{(l)T}\frac{\partial J}{\partial\mathbf{Z}^{(l)}}
$$

Note that the cost gradients for the output layer (the last layer) are implemented in the `Network` class.

Below is an illustration of how the above thread of equations for back propagation play out:

Say we are computing the gradients of 3rd layer of a network, i.e. $$l=3$$. This means we must have already finished computing the gradients for the 4th layer, because in backpropagation you start at the output layer (the last layer) and work your way back to the input layer. This means during the procedure for the 4th layer, we ran this computation:

$$
\frac{\partial J}{\partial\mathbf{A}^{\left(3\right)}}=\mathbf{W}^{(4)T}\frac{\partial J}{\partial\mathbf{Z}^{(4)}}
$$

So, when we begin the computations for layer 3, starting with $$\frac{\partial J}{\partial\mathbf{Z}^{\left(l\right)}}$$, we already have $$\frac{\partial J}{\partial\mathbf{A}^{\left(3\right)}}$$:

$$
\frac{\partial J}{\partial\mathbf{Z}^{\left(3\right)}}=\frac{\partial J}{\partial\mathbf{A}^{\left(3\right)}}\odot f^\prime\left(\mathbf{Z}^{\left(3\right)}\right)
$$

```python
def _layer_back_prop(self):

	Z = self.Z
	A = self.A
	
	self.gradients["dAdZ"]=self.activation_type._backward_pass(A, Z)

	self.gradients["dJdZ"] =self.gradients["dJdA"] * self.gradients["dAdZ"]
	
	self._compute_cost_gradients()
	
	return None
	
def _compute_cost_gradients(self):
	
	A_prior = self.preceding_layer.A

	self.gradients["dJdB"] = np.sum(self.gradients["dJdZ"], axis=1) 
	
	self.gradients["dJdW"] = np.matmul(self.gradients["dJdZ"], A_prior.T)
	
	self.preceding_layer.gradients["dJdA"] = np.matmul(self.W.T, self.gradients["dJdZ"])
	
	return None
```

#### **Input Layer**

The `_InputLayer` is a subclass of `Layer` and represent the input layer of the network. It’s a layer whose activations are the input data, and its serial position in the network is 0 (i.e. it is layer 0).

```python
class _InputLayer(Layer):
    def __init__(self, parent_network):
        super().__init__(activation_name=None, num_units= None)
        self.position_in_network=0
        self.parent_network=parent_network
    def _populate(self, X):
        self.A=X
        self.num_units=X.shape[0]
```

### **Parameter Initializer**

The `ParameterInitializer` class enables us to choose how we want to initialize the parameters in our network.

Two schemes are implemented for the initialization of weights. The first scheme initializes the weights with values randomly generated from a standard normal distribution, and it also scales down the values to very small numbers by multiplying them with `self.factor`.

The second scheme is the Xavier initialization scheme [first introduced](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) in 2010 by Xavier Glorot. In this scheme, we initialize the network in such a way that the variance of the activations are constant across all layers, and to do that we use equations 10 and 11 from the linked paper (although equation 12 is probably better).

For the biases, we either initialize them all to 0 or all to 1.

```python
class ParameterInitializer:

    def __init__(self, weight_init_scheme="xavier", bias_init_scheme="zeros", 
                 factor=0.01,random_seed=3):
        
        self.weight_init_scheme=weight_init_scheme
        self.bias_init_scheme=bias_init_scheme
        self.factor=factor
        self.rnd=np.random.RandomState(random_seed)
        self.status=0
        self.parent_network=None
        
    def _execute_initialization_if_notdone(self):
        
        if self.status==0:
            L = self.parent_network.num_layers
        
            for l in range(1, L+1):
                
                # initializing weights.
                self._initialize_weights(l)
                
                # initializing biases.
                self._initialize_biases(l)
                
            self.status=1
        
        return None
    
    def _initialize_weights(self, layer_sn):
        l=layer_sn
        if self.weight_init_scheme=="default":
            self.parent_network.layers[l].W = self.rnd.randn(self.parent_network.layers[l].num_units, 
                                              self.parent_network.layers[l-1].num_units)\
                * self.factor
        
        elif self.weight_init_scheme=="xavier":
            self.parent_network.layers[l].W = self.rnd.randn(self.parent_network.layers[l].num_units, 
                                              self.parent_network.layers[l-1].num_units)\
                / np.sqrt(self.parent_network.layers[l-1].num_units)
                
        return None
        
    def _initialize_biases(self, layer_sn):
        l=layer_sn
        if self.bias_init_scheme=="zeros":
            self.parent_network.layers[l].B = np.zeros((self.parent_network.layers[l].num_units, 1))\
                * self.factor
        elif self.bias_init_scheme=="ones":
            self.parent_network.layers[l].B = np.ones((self.parent_network.layers[l].num_units, 1))\
                * self.factor              

        return None
```

Note that the parameters are not actually initialized when an instance of `ParameterInitializer` is created, but instead just before forward propagation when the method `_execute_initialization_if_notdone` is invoked. The `ParameterInitializer` instance retains information passed to it for initialization, but never performs it until forward propagation.

### **Activation Fucntion**
The `_ActivationFunction` class is where all activation functions, $$f\left(Z^{\left(l\right)}\right)$$, and their derivatives, $$f'\left(Z^{\left(l\right)}\right)$$, are implemented.

In the code, $$f'\left(Z^{\left(l\right)}\right)$$ is named `dAdZ`. But in a strict sense that name is a misnomer because $$f'\left(Z^{\left(l\right)}\right)$$ is not exactly equal to $$\frac{\partial\mathbf{A}^{(l)}}{\partial\mathbf{Z}^{\left(l\right)}}$$, but they are related because in a single artificial neuron model it is the diagonal of  $$\frac{\partial\vec{a}}{\partial\vec{z}}$$ (see [parts 3](/optimize-an-artificial-neuron-from-scratch.html#consolidate-the-results) of this blog series). Maybe I should’ve just used a name that looks more like $$f'\left(Z^{\left(l\right)}\right)$$.

For activation functions, I found [this list on Wikipedia](https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions) very useful. Pick an activation function and its derivative, verify them, and then implement them.

```python
class _ActivationFunction: # module-private.
    
    _available_activation_funcs=["logistic","relu","tanh", "linear"]
    
    def __init__(self, name):      
        if any(a == name.lower() for a in self.__class__._available_activation_funcs):
            self.name=name.lower()
        else:
            raise ValueError
        
    def _forward_pass(self, Z): # module-private.
        if self.name=="logistic":
            A = self._logistic(Z)
        if self.name=="relu":
            A = self._relu(Z)
        if self.name=="tanh":
            A = self._tanh(Z)
        if self.name=="linear":
            A = self._linear(Z)
        return A
    
    def _backward_pass(self, A,Z): # module-private.
        if self.name=="logistic":
            dAdZ=self._logistic_gradient(A)
        if self.name=="relu":
            dAdZ=self._relu_gradient(Z)
        if self.name=="tanh":
            dAdZ=self._tanh_gradient(Z)
        if self.name=="linear":
            dAdZ = self._linear_gradient(Z)    
        return dAdZ
    
    def _logistic(self, Z): # class-private.
        A = 1/(1+np.exp(-Z))
    
        return A

    def _relu(self, Z): # class-private.
        A = np.maximum(0,Z)
    
        return A
    
    
    def _tanh(self, Z): # class-private.
        A=(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        return A
    
    def _linear(self, Z): # class-private.
        return Z
    
    def _relu_gradient(self, Z): # class-private.
        result = np.array(Z, copy=True)
    
        result[Z <= 0] = 0
        result[Z > 0] = 1
        
        dAdZ=result
        return dAdZ
    
    def _logistic_gradient(self, A): # class-private.

        dAdZ = A * (1-A)
    
        return dAdZ
    
    def _tanh_gradient(self, A): # class-private.

        dAdZ=1-A**2
        
        return dAdZ
        
    def _linear_gradient(self, Z): # class-private.
        return np.ones(Z.shape)
```

### **Network**

A `Network` instance represents a network. It holds everything related to the network, which includes attributes pointing to a collection of all the Layer instances that have been assigned to the network, the most recently computed performance metrics (cost, accuracy, precision, etc.) for the network, etc.

It also has an attribute pointing to the training archiver of the network (i.e. a TrainingArchiver instance) and another pointing to the parameter initializer of the network (i.e. a ParameterInitializer instance).

The `Network` object computes the gradients for the output layer (last layer) and controls backpropagation across the entire network by invoking each Layer instance that belongs to it to compute the layer’s gradients.

The equation for the output layer gradient, $$\frac{\partial J}{\partial\mathbf{A}^{(L)}}$$, implemented in the `Network` class is:

$$
\frac{\partial J}{\partial\mathbf{A}^{(L)}}=-\frac{1}{m}\bullet\left(\frac{\vec{y}}{\mathbf{A}^{(L)}}-\frac{1-\vec{y}}{1-\mathbf{A}^{(L)}}\right)
$$

```python
def _compute_last_layer_dJdA(self):

	L=self.num_layers
	m = self.Y_batch.shape[1]
	A_last=self.layers[L].A
	
	self.layers[L].gradients["dJdA"] = -(1/m) * ((self.Y_batch / A_last) - 
								((1 - self.Y_batch) / (1 - A_last)))
	return None


def _network_back_prop(self): # class-private.

	L = self.num_layers
	
	# Initialize the backpropagation by computing dJ/dA of the last layer (Lth layer).
	self._compute_last_layer_dJdA()
		
	# Compute the Lth layer gradients.
	last_layer = self.layers[L]
	last_layer._layer_back_prop()
	
	# ensure dJdB is a 2D numpy array and not 1D, even though it stored as a vector, 
	# and only broadcasted into a matrix during computations.
	last_layer.gradients["dJdB"] = last_layer.gradients["dJdB"].reshape(-1,1)
	
	# Compute the gradients of the other layers.
	for l in reversed(range(1, L)):
		current_layer = self.layers[l]
		
		current_layer._layer_back_prop()
				
		# ensure dJdB is a 2D numpy array and not 1D.
		current_layer.gradients["dJdB"] = current_layer.gradients["dJdB"].reshape(-1,1)  

	return None
```

To repurpose the network for regression, we just need to replace cross entropy with mean squared error for our cost function. Then the result of the output layer gradient, $$\frac{\partial J}{\partial\mathbf{A}^{(L)}}$$, will also change.

The network also computes the performance metrics, like cost, accuracy and precision.

```python
def _compute_cost(self): # module-private.
	L=self.num_layers
	m = self.Y_batch.shape[1] # number of records/instances/examples.
	A_last=self.layers[L].A
	
	# Computes cross entropy loss. The equation assumes both A_last and Y_batch are vectors (binary classification).
	self.cost = (-1/m) * np.sum((self.Y_batch * np.log(A_last)) + 
										  ((1 - self.Y_batch) * np.log(1 - A_last)))
	
	self.cost = np.squeeze(self.cost) # ensures cost is a scalar (this turns [[10]] or [10] into 10).
		
	return None

def _compute_accuracy(self): # module-private.
	Y_true=self.Y_batch.reshape(-1,)
	Y_pred=self.Y_pred.reshape(-1,)
	
	# assumes binary classification.
	self.latest_accuracy=np.average(np.where(Y_true==Y_pred, 1,0))
	
	return None

def _compute_precision(self): # module-private.
	Y_true=self.Y_batch.reshape(-1,)
	Y_pred=self.Y_pred.reshape(-1,)
	
	# assumes binary classification.
	mask_pred_positives = (Y_pred==1)
	self.latest_precision=np.average(np.where(Y_pred[mask_pred_positives]==Y_true[mask_pred_positives], 1, 0))
	
	return None
```

The network performs parameter updating via vanilla gradient descent. Vanilla means that there are no improvements like momentum, adaptive learning rates, etc. This is stochastic, mini-batched. The stochastic batching is taken care of inside the train method. 

```python
def _update_parameters_gradient_descent(self, learning_rate): # class-private.

	L = self.num_layers # number of layers in the network (also sn of last layer).

	# the basic gradient descent.
	for l in range(1, L+1):
		self.layers[l].W = self.layers[l].W - learning_rate * self.layers[l].gradients["dJdW"]
		self.layers[l].B = self.layers[l].B - learning_rate * self.layers[l].gradients["dJdB"]        
	
	return None
```

The network also has a convenience method for initializing its parameters. 

```python
    def initialize_parameters(self, weight_init_scheme="xavier", bias_init_scheme="zeros", 
                              factor=0.01,random_seed=3): # public.
        if (factor<=0 or factor>=1):
            raise ValueError("factor must range from 0 to 1.")
            
        weight_init_scheme = weight_init_scheme.lower()
        bias_init_scheme = bias_init_scheme.lower()
                
        self.parameter_initializer=ParameterInitializer(weight_init_scheme=weight_init_scheme, 
                                                         bias_init_scheme=bias_init_scheme,
                                                         factor=factor,random_seed=random_seed)
        self.parameter_initializer.parent_network=self
```

The above method simply creates the parameter initializer (a ParameterInitializer instance) for the network. But the network does not actually initialize the parameters, because there is no guarantee on when the network will know the number of units in the input layer, which is needed for parameter initialization. It is possible that information will be unavailable until the first forward pass. As such, when the `parameter_initializer` (the parameter initializer of the network) is created, it simply stores the information needed for initialization, and the actual parameter initialization only occurs when the `parameter_initializer` invokes the method `_execute_initialization_if_notdone` during forward propagation.

The network also takes care of the training loop, which involves the following:

1.	Checks that the network is ready for training: at least one layer has been added to the network, a parameter initializer has been specified (via the method initialize_parameters), and a training archive has been added.
2.	Checks that the input data and the target have the right shapes.
3.	Run one iteration of training, which involves:
  * Randomly samples a batch from the input data and target.
  * Performs forward propagation with that batch.
  * Performs back propagation.
  * Update parameters via gradient descent.
  * Compute performance metrics (cost, accuracy, etc.) and use the training archiver to cache them.
4.	Repeat 3 step until we reach the specified number of iterations.

```python
def train(self, X, Y, num_iterations=10, batch_size=None, learning_rate=0.0000009, print_start_end=True,
		  validation_X=None, validation_Y=None): 
	
	self._check_readiness_to_train() # Raises an exception if not ready.
	
	if (Y.shape[1]!=X.shape[1] or Y.shape[0]!=1):
		raise ValueError("X and Y must have compatible shapes, n x m and 1 x m respectively.")
		
	if (Y.shape[0] != self.layers[self.num_layers].num_units):
		raise ValueError("Y and the output layer must have compatible shapes.")
	
	self.Y=Y 
	self.X=X         
	
	if print_start_end==True: print("Training Begins...")

	num_iterations=num_iterations+self.num_latest_iteration
	
	for self.num_latest_iteration in range(self.num_latest_iteration+1, num_iterations+1):
		# loop header: allows training to resume from the previous state of the network
		#  at end of its last training if any.
			
		# select batch from the training dataset.
		if batch_size is None: batch_size=self.X.shape[1] # use the entire data.
		
		random_indices = np.random.choice(self.Y.shape[1], (batch_size,), replace=False)
		
		self.Y_batch=self.Y[:,random_indices]
		self.input_layer._populate(self.X[:,random_indices])
			  
		# Forward propagation.
		self._network_forward_prop()
	
		# Back propagation.
		self._network_back_prop()
	 
		# Update parameters.
		self._update_parameters_gradient_descent(learning_rate=learning_rate)
		
		# Compute the training cost, accuracy and precision (using the current training batch).
		
		self.training_archiver._compute_and_archive_cost(cost_type="training")
		self.training_archiver._compute_and_archive_accuracy(acc_type="training")
		self.training_archiver._compute_and_archive_precision(precis_type="training")
		
		# Compute the validation cost, accuracy and precision (using validation dataset).
		if (validation_Y is None) or (validation_X is None):
			pass
		else:
			self.input_layer._populate(validation_X)
			self.Y_batch = validation_Y
			
			self._network_forward_prop()
			self.training_archiver._compute_and_archive_cost(cost_type="validation")
			self.training_archiver._compute_and_archive_accuracy(acc_type="validation")
			self.training_archiver._compute_and_archive_precision(precis_type="validation")
		
		# rest of caching occurs here.
		
		self.training_archiver._archive_gradients()
		self.training_archiver._archive_parameters()
		
		# print archiving messages if any:
		if self.training_archiver.report: 
			self.training_archiver._print_report()
			self.training_archiver._clear_report()
		
	if print_start_end: print("Training Complete!")
```

The network also has an evaluate method that simply performs forward propagation on an input batch of data and returns the resulting value for the specified performance metric.

### **Training Archiver**
The `TrainingArchiver` class enables us to set which components of the network and performance metric to cache and display during training. Nothing really technical here. The code speaks for itself.

```python
class TrainingArchiver:

    archival_targets= ["activation", "preactivation", "cost", "gradient", 
                    "parameters", "accuracy", "precision"]
    num_archival_targets=len(archival_targets)
    
    def __init__(self, broad_frequency=None):

        self.archiving_frequencies=dict(
            zip(self.__class__.archival_targets,[broad_frequency]*self.__class__.num_archival_targets))
        
        self.archiving_verbosities=dict(
            zip(self.__class__.archival_targets,[0]*self.__class__.num_archival_targets))
        
        self.all_gradients = dict()
        self.all_parameters = dict()
        self.all_preactivations = dict()
        self.all_activations = dict()
        
        self.all_training_accuracies = dict()
        self.all_validation_accuracies = dict()
        self.all_training_precisions = dict()
        self.all_validation_precisions = dict()
        self.all_training_costs = dict()
        self.all_validation_costs = dict()
        
        self.target_network=None
        
        self.report=""
                
    def set_archiving_frequencies(self, **kwargs): # public.
        
        self.archiving_frequencies["activation"]=kwargs.get("activation", 0)
        self.archiving_frequencies["preactivation"]=kwargs.get("preactivation", 0)
        self.archiving_frequencies["cost"]=kwargs.get("cost", 0)
        self.archiving_frequencies["gradient"]=kwargs.get("gradient", 0)
        self.archiving_frequencies["parameters"]=kwargs.get("parameters", 0)
        self.archiving_frequencies["accuracy"]=kwargs.get("accuracy", 0)
        self.archiving_frequencies["precision"]=kwargs.get("precision", 0)
    
    def set_archiving_verbosities(self, **kwargs): # public.

        self.archiving_verbosities["activation"]=kwargs.get("activation", 0)
        self.archiving_verbosities["preactivation"]=kwargs.get("preactivation", 0)
        self.archiving_verbosities["cost"]=kwargs.get("cost", 0)
        self.archiving_verbosities["gradient"]=kwargs.get("gradient", 0)
        self.archiving_verbosities["parameters"]=kwargs.get("parameters", 0)
        self.archiving_verbosities["accuracy"]=kwargs.get("accuracy", 0)
        self.archiving_verbosities["precision"]=kwargs.get("precision", 0)
        
    def _set_target_network(self, target_network): # module-private.
        self.target_network=target_network
    
    def _archive_activations(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["activation"]!=0) and (i % self.archiving_frequencies["activation"] == 0):
            L = self.target_network.num_layers
            acts_all_layers=dict()
            for l in range(1, L+1):
                acts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].A)
            self.all_activations[i]=acts_all_layers
    
    def _archive_preactivations(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["preactivation"]!=0) and (i % self.archiving_frequencies["preactivation"] == 0):
            L = self.target_network.num_layers
            preacts_all_layers=dict()
            for l in range(1, L+1):
                preacts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].Z)
            self.all_preactivations[i]=preacts_all_layers        
        
    def _archive_gradients(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["gradient"]!=0) and (i % self.archiving_frequencies["gradient"] == 0):
            L = self.target_network.num_layers
            grads_all_layers=dict()
            for l in range(1, L+1):
                grads_all_layers[l]=copy.deepcopy(self.target_network.layers[l].gradients)
            self.all_gradients[i]=grads_all_layers
    
    def _archive_parameters(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["parameters"]!=0) and (i % self.archiving_frequencies["parameters"] == 0):
            L = self.target_network.num_layers
            params_all_layers=dict()
            for l in range (1, L+1):
                params_all_layers["W"+str(l)]=copy.deepcopy(self.target_network.layers[l].W)
                params_all_layers["B"+str(l)]=copy.deepcopy(self.target_network.layers[l].B)
            self.all_parameters[i]=params_all_layers
            
    def _compute_and_archive_accuracy(self, acc_type): # module-private.
        i = self.target_network.num_latest_iteration
        if (self.archiving_frequencies["accuracy"]!=0) and (i % self.archiving_frequencies["accuracy"] == 0):
            if acc_type=="training":
                self.target_network._compute_accuracy()
                self.all_training_accuracies[i]=self.target_network.latest_accuracy
                self._update_report(archival_target="accuracy", prefix="training")
            if acc_type=="validation":
                self.target_network._compute_accuracy()
                self.all_validation_accuracies[i]=self.target_network.latest_accuracy
                self._update_report(archival_target="accuracy", prefix="validation")
                
    def _compute_and_archive_precision(self, precis_type): # module-private.
        i = self.target_network.num_latest_iteration
        if (self.archiving_frequencies["precision"]!=0) and (i % self.archiving_frequencies["precision"] == 0):
            if precis_type=="training":
                self.target_network._compute_precision()
                self.all_training_precisions[i]=self.target_network.latest_precision
                self._update_report(archival_target="precision", prefix="training")
            if precis_type=="validation":
                self.target_network._compute_precision()
                self.all_validation_precisions[i]=self.target_network.latest_precision
                self._update_report(archival_target="precision", prefix="validation")
                
    def _compute_and_archive_cost(self, cost_type): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["cost"]!=0) and (i % self.archiving_frequencies["cost"] == 0):
            
            if cost_type=="training":
                self.target_network._compute_cost()
                self.all_training_costs[i] = self.target_network.cost
                self._update_report(archival_target="cost", prefix="training")
                
            if cost_type=="validation":
                self.target_network._compute_cost()
                self.all_validation_costs[i] = self.target_network.cost
                
                self._update_report(archival_target="cost", prefix="validation")

    def _update_report(self, archival_target, prefix=None, suffix=None): # module-private.
        if prefix==None: prefix=""
        if suffix==None: suffix=""
        
        i = self.target_network.num_latest_iteration
        
        if self.archiving_verbosities[archival_target]:
            
            if archival_target=="accuracy" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_validation_accuracies[i]))+"\n"
            elif archival_target=="accuracy" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_training_accuracies[i]))+"\n"                
            
            if archival_target=="cost" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_validation_costs[i]))+"\n"
            elif archival_target=="cost" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_training_costs[i]))+"\n"

            if archival_target=="precision" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_validation_precisions[i]))+"\n"
            elif archival_target=="precision" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_training_precisions[i]))+"\n"
    def _clear_report(self):
        self.report=""
    def _print_report(self):
        print(self.report,"="*10)
```

## **Some Opportunities to extend this exercise**

This blogseries went over the foundations of deep learning. But in the scale of things, it is a small beginning in the rapidly expanding field of deep learning, as the field has grown far beyond the standard feedforward neural network. As such, the opportunities for extending and improving the neural network showcased in this blogseries are endless.

However, it is advisable to use a machine learning framework that gives you enough flexibility to put together a neural net piece by piece. NumPy is not that. One excellent alternative is TensorFlow.

Here are some of the things we can do to expand or improve the neural net.

### **Multiclass classification**

In this blogseries we imposed certain limitations on the design of our feedforward neural network. We limited the output layer to one node, which means we can only handle regression and binary classification tasks.

Multiclass classification requires that our output layer take more than one node, so we can relax our assumptions to accommodate that. This means the equations that we developed for the gradients of the output layer with have to be reworked. It means we also have to introduce the softmax activation function, which the activation function of choice for multiclass classification. It’s a function that generalizes the logistic function to multiclass situation.

### **Improvements to cost function**

There are several improvements that we can make to the cost function. We can introduce a regularization term to our cost function in order to prevent certain pathways in the network from heavily dominating the others. This usually helps improve generalization. We can also introduce class weights to help counterbalance the adverse impact of class imbalance in the training set. Class weights enable the cost function to bring balance to the magnitude of cost contributed by each class in any batch of dataset.

### **Other parameter updating schemes**

One way to improve the updating for the network’s parameters is to use a scheme that’s based on the stochastic gradient descent (SGD) but with an adaptive learning rates (instead of a fixed one) and or momentum of the gradients. Some examples of such updating schemes are SGD with Nesterov momentum, adaptive moment estimation (abbrv. Adam), Adagrad, etc.
