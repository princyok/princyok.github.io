---
layout: post
title:  "Catching AI with its pants down: Understand a Neural Network from Scratch"
logline: "Exploring the mathematics of a neural network."
date:   "2020-04-27"
categories: machine-learning
permalink:
comments: true
---
{% include scripts.html %}

{% include blogseries_mantra_catching_ai.html %}

* TOC
{:toc}

## **Prologue**

This is part 5 of this blog series, *Catching AI with its pants down*. This blog series aims to explore the inner workings of neural networks and show how to build a standard feedforward neural network from scratch.

In this part, I will go over the mathematical underpinnings of a standard feedfoward neural network.

{% include blogseries_index_catching_ai.html %}


## **Network of artificial neurons**
A neural network is a network of artificial neurons. They are basically function approximators, and according to the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem){:target="_blank"}, are capable of approximating continuous functions to any degree of accuracy. While that might sound very profound, the theorem doesn’t tell us how to optimize the neural network, so it is not a panacea.

We’ve seen how to build a single artificial neuron in part 1 to 4. We set up our equations entirely as equations of tensors, which makes transitioning to neural networks significantly easier.

In a neural network, the units (neurons) are stacked in layers, where outputs from the previous layer serve as the inputs to the next layer. Activations get propagated forward through the network, hence the name forward propagation (or simply forward pass). Errors get propagated backward through the network, hence the name backward propagation. The process of back propagation of errors and updating of weights are together commonly referred to as backward pass.

The input to the neural network is the data, and it is commonly referred to as the input layer. It may be the preprocessed data instead of the raw data. The input layer is not counted as the first layer of the network and is instead counted as the 0th layer if counted at all. 

The last layer is known as the output layer and its output is the prediction or estimation. Every other layer in the network, which is any layer between the input and output layers, is called a hidden layer. This also means that the first hidden layer is also the first layer of the network.

For instance, a five-layer feedforward neural network has one input layer, four hidden layers and finally an output layer.

{% include image.html url='/assets/images/neural_network/fully_connected_standard_feedforward_neural_network.png' description='A generalized depiction of a fully connected standard feedforward neural network, also ambiguously referred to as a multilayer perceptron. It is fully connected because each unit is connected to all units in the next layer. It is feedforward because all the connections are moving only forward, e.g. there are no loops.' %}

Neural networks come in an assortment of architectures. As an introduction, we are focusing on the standard feedforward neural network, also commonly referred to as multilayer perceptron, even though the neurons may not necessarily be the original perceptron with Heaviside activation function. The term feedforward indicates that our network has no loops or cycles; that is, the activations do no loop backward or within the layer but are always flowing forward. In contrast, there are architectures were activations cycle around, e.g. recurrent neural networks.

If you’re curious what other architecture exist out there, check out [this beautiful article](https://www.asimovinstitute.org/neural-network-zoo/){:target="_blank"}.

## **Forward propagation**

Now, let’s consider a three-layer multilayer perceptron, with two units in the first hidden layer, another two in the second hidden layer, and one unit in the output layer.

{% include image.html url="/assets/images/neural_network/three_layer_MLP.png" description="A three-layer multilayer perceptron." %}

The notations being used here are as follows:

{% include indent_paragraph.html content="
The activations are written as $ a_i^{(l)} $, where $ l $ is the serial number of the layer, and $ i $ is the serial number of the unit in the $ l $<sup>th</sup> layer. E.g. for the second unit of the second layer, it is $ a_2^{(2)} $.
<br><br>
The weight for each connection is denoted as $ w_{i,\ \ h}^{(l)} $, where $ l $ is the serial number of the layer (it always is), $ i $ is the serial number of the unit in the $ l $<sup>th</sup> layer, which is the destination of the connection, and $ h $ is the serial number of the unit in the $ (l-1) $<sup>th</sup> layer, which the origin of the connection.
<br><br>
As an example, $ w_{1,2}^{(3)} $ is the weight of the connection pointing from the 2<sup>nd</sup> unit of the 2<sup>nd</sup> layer (the unit with activation $ a_2^{(2)} $) to the 1<sup>st</sup> unit of the 3<sup>rd</sup> layer (the unit with activation $ a_1^{(3)} $).
<br><br>
Note that whenever we are focused on a specific layer, let's designate it as the current layer, it will be regarded as layer $ l $, and the layer before it will be layer $ l-1 $, while the layer after it will be layer $ l+1 $.
<br><br>

Just like in every article for this blog series, unless explicitly started otherwise, matrices (and higher-order tensors, which should be rare in this blog series) are denoted with boldfaced, non-italic, uppercase letters (e.g. $ \mathbf{A}^{(l)} $ for the matrix that holds all activations in layer $ l $); vectors are denoted with non-boldfaced, italic letters accented with a right arrow; and scalars are denoted with non-boldfaced, italic letters.
<br><br>
Other notations will be introduced along the way.
"%}

The input layer can be treated as the 0<sup>th</sup> layer, where $$x_i$$ becomes denoted as $$a_i^{(0)}$$. This means the number of features now becomes the number of units in the input layer. The bias node, which is usually never shown and always equal to 1, is treated as the 0<sup>th</sup> unit (node) in each layer .

{% include image.html url="/assets/images/neural_network/three_layer_MLP_input_layer_activations.png" description="A three-layer multilayer perceptron with the input treated as activations of the input layer." %}


Since we’ve thoroughly gone over how to mathematically characterize a single unit in part 2, we will use that as the springboard for delineating the three-layer network. Let’s focus on one of the units, say the second unit of the second layer.

{% include image.html url="/assets/images/neural_network/second_unit_second_layer_in_three_layer_MLP.png" description="Focusing on the second unit of the second layer of the three-layer multilayer perceptron, as well as the units and connections feeding into it." %}

The inputs to the second unit of the second layer are the activations produced by the units of the first layer. We’ve seen exactly this in part 2, except that here our input to the second unit are activations ($$\mathbf{A}^{(1)}$$) instead of the data ($$\mathbf{X}$$). 

Therefore, our equation of tensors is going to be:

$$
{\vec{a}}_2^{(2)}=f\left({\vec{z}}_2^{(2)}\right)
$$

$$
{\vec{z}}_2^{(2)}=\ {\vec{w}}_2^{(2)}\mathbf{A}^{(1)}\ +\ {\vec{b}}_2^{(2)}
$$

{% include indent_paragraph.html content="
Where $ {\vec{z}}_2^{(2)} $ and $ {\vec{a}}_2^{(2)} $, are the preactivations and activations of for the second unit in the second layer, and both have the same shape 1-by-$ m $, where $ m $ is the number of examples in the batch.
<br><br>
And $ {\vec{w}}_2^{(2)} $ contains the weights for the connections pointing to the second unit of the second layer, and it has the shape 1-by-$ n^{(2)} $, where $ n^{(2)} $ is the number of units in the second layer (layer 2).

$$
{\vec{w}}_2^{(2)}=\left[\begin{matrix}w_{2,\ 1}^{(2)}&w_{2,\ \ 2}^{(2)}\\\end{matrix}\right] 
$$
And as in part 2, $ {\vec{b}}_2^{(2)} $ is fundamentally a scalar but gets broadcasted to match the shape of $ {\vec{z}}_2^{(2)} $ during computation.
<br><br>
And $ \mathbf{A}^{(1)} $ contains the activations from the first layer and has shape $ n^{(1)} $-by-$ m $, where $ n^{(1)} $ is the number of units in the first layer (layer 1).
"%}

This gives us a template to write out the equations for the other units in the network. For example, for the first unit in the second layer, we have:

$$
{\vec{a}}_1^{(2)}=f\left({\vec{z}}_1^{(2)}\right)
$$

$$
{\vec{z}}_1^{(2)}=\ {\vec{w}}_1^{(2)}\mathbf{A}^{(1)}\ +\ {\vec{b}}_1^{(2)}
$$

If we had more units in the second layer, the equation would be:

$$
{\vec{a}}_i^{(2)}=f\left({\vec{z}}_i^{(2)}\right)
$$

$$
{\vec{z}}_i^{(2)}=\ {\vec{w}}_i^{(2)}\mathbf{A}^{(1)}\ +\ {\vec{b}}_i^{(2)}
$$

Notice that we can now write the equations for the preactivations for the second layer:

$$
{\vec{z}}_1^{(2)}=\ {\vec{w}}_1^{(2)}\mathbf{A}^{(1)}\ +\ {\vec{b}}_1^{(2)}
$$

$$
z_2^{(2)}=\ {\vec{w}}_2^{(2)}\mathbf{A}^{(1)}\ +\ {\vec{b}}_2^{(2)}
$$

$$\vdots$$

$$
{\vec{z}}_{n^{(2)}}^{(2)}=\ {\vec{w}}_{n^{(2)}}^{(2)}\mathbf{A}^{(1)}\ +\ {\vec{b}}_{n^{(2)}}^{(2)}
$$

Just like in part 2, we can put the above system of equations into matrix format. One caveat is to remember that the terms in the above equations are themselves vectors and matrices, so we use relationship between matrix-matrix and vector-matrix multiplications:

$$
\left[\begin{matrix}{\vec{z}}_1^{(2)}\\{\vec{z}}_2^{(2)}\\\vdots\\{\vec{z}}_{n^{(2)}}^{(2)}\\\end{matrix}\right]=\left[\begin{matrix}{\vec{w}}_1^{(2)}\mathbf{A}^{(1)}\\{\vec{w}}_2^{(2)}\mathbf{A}^{(1)}\\\vdots\\{\vec{w}}_{n^{(2)}}^{(2)}\mathbf{A}^{(1)}\\\end{matrix}\right]+\left[\begin{matrix}{\vec{b}}_1^{(2)}\\{\vec{b}}_2^{(2)}\\\vdots\\{\vec{b}}_{n^{(2)}}^{(2)}\\\end{matrix}\right]
$$

$$
\mathbf{Z}^{(2)}=\mathbf{W}^{(2)}\mathbf{A}^{(1)}+\mathbf{B}^{(2)}
$$

Or in general, for a feedforward neural network:

$$
\left[\begin{matrix}{\vec{z}}_1^{(l)}\\{\vec{z}}_2^{(l)}\\\vdots\\{\vec{z}}_{n^{(l)}}^{(l)}\\\end{matrix}\right]=\left[\begin{matrix}{\vec{w}}_1^{(l)}\mathbf{A}^{(l-1)}\\{\vec{w}}_2^{(l)}\mathbf{A}^{(l-1)}\\\vdots\\{\vec{w}}_{n^{(l)}}^{(l)}\mathbf{A}^{(l-1)}\\\end{matrix}\right]+\left[\begin{matrix}{\vec{b}}_1^{(l)}\\{\vec{b}}_2^{(l)}\\\vdots\\{\vec{b}}_{n^{(l)}}^{(l)}\\\end{matrix}\right]
$$

$$
\mathbf{Z}^{(l)}=\mathbf{W}^{(l)}\mathbf{A}^{(l-1)}+\mathbf{B}^{(l)}
$$

Our preactivation tensor $$\mathbf{Z}^{(l)}$$ is an $$n^{(l)}$$-by-$$m$$ matrix of the form:

$$
\mathbf{Z}^{(l)}=\left[\begin{matrix}{\vec{z}}_1^{(l)}\\{\vec{z}}_2^{(l)}\\\vdots\\{\vec{z}}_{n^{(l)}}^{(l)}\\\end{matrix}\right]=\left[\begin{matrix}z_{1,1}^{(l)}&z_{1,2}^{(l)}&\cdots&z_{1,m}^{(l)}\\z_{2,1}^{(l)}&z_{2,2}^{(l)}&\cdots&z_{2,m}^{(l)}\\\vdots&\vdots&\ddots&\vdots\\z_{n^{(l)},1}^{(l)}&z_{n^{(l)},2}^{(l)}&\cdots&z_{n^{(l)},m}^{(l)}\\\end{matrix}\right]
$$

The activation tensor $$\mathbf{A}^{(l-1)}$$ is an $$n^{(l-1)}$$-by-$$m$$ matrix. In general, the activation for any layer $$l$$, denoted $$\mathbf{A}^{(l)}$$, has the shape $$n^{(l)}$$-by-$$m$$.

$$
\mathbf{A}^{(l)}=\left[\begin{matrix}{\vec{a}}_1^{(l)}\\{\vec{a}}_2^{(l)}\\\vdots\\a_{n^{(l)}}^{(l)}\\\end{matrix}\right]=\left[\begin{matrix}a_{1,1}^{(l)}&a_{1,2}^{(l)}&\cdots&a_{1,m}^{(l)}\\a_{2,1}^{(l)}&a_{2,2}^{(l)}&\cdots&a_{2,m}^{(l)}\\\vdots&\vdots&\ddots&\vdots\\a_{n^{(l)},1}^{(l)}&a_{n^{(l)},2}^{(l)}&\cdots&a_{n^{(l)},m}^{(l)}\\\end{matrix}\right]
$$

The weight tensor $$\mathbf{W}^{(l)}$$ is an $$n^{(l)}$$-by-$$n^{(l-1)}$$ matrix, where $$n^{(l)}$$ and $$n^{(l-1)}$$ are number of units in the $$l$$<sup>th</sup> and $$(l-1)$$<sup>th</sup> layers.

$$
\mathbf{W}^{(l)}=\left[\begin{matrix}{\vec{w}}_1^{(l)}\\{\vec{w}}_2^{(l)}\\\vdots\\{\vec{w}}_{n^{(l)}}^{(l)}\\\end{matrix}\right]=\left[\begin{matrix}w_{1,1}^{(l)}&w_{1,2}^{(l)}&\cdots&w_{1,n^{(l-1)}}^{(l)}\\w_{2,1}^{(l)}&w_{2,2}^{(l)}&\cdots&z_{2,n^{(l-1)}}^{(l)}\\\vdots&\vdots&\ddots&\vdots\\w_{n^{(l)},1}^{(l)}&w_{n^{(l)},2}^{(l)}&\cdots&w_{n^{(l)},n^{(l-1)}}^{(l)}\\\end{matrix}\right]
$$

There is a reason why each of the above matrices is also presented as a column vector of row vectors. This kind of representation will come in very handy when we try to solve the cost gradients symbolically via back propagation.

And as you probably already expect, the full forward propagation equation for a feedforward neural network is:

$$
\mathbf{A}^{(l)}=f(\mathbf{Z}^{\left(l\right)})
$$

$$
\mathbf{Z}^{(l)}=\mathbf{W}^{(l)}\mathbf{A}^{(l-1)}+\mathbf{B}^{(l)}
$$

The way it works is we use our data (maybe after some preprocessing, like scaling the data, binning it, etc.) as the activations of the input layer $$\mathbf{A}^{(0)}$$ to compute $$\mathbf{Z}^{(1)}$$, and then use that to get $$\mathbf{A}^{(1)}$$. Then we use $$\mathbf{A}^{(1)}$$ to compute $$\mathbf{A}^{(2)}$$. Then we use $$\mathbf{A}^{(2)}$$ to compute $$\mathbf{Z}^{(3)}$$ and subsequently $$\mathbf{A}^{(3)}$$. We continue the pattern until we finally compute the activations of the output layer $$\mathbf{A}^{(L)}$$ (where $$L$$ is the serial number of the last layer).

It is quickly evident that the shape of $$\mathbf{A}^{(\mathbf{0})}$$ is determined by the data, in particular by the number of features ($$\mathbf{X}$$), which becomes the number of units in the input layer. Similarly, the shape of $$\mathbf{A}^{(L)}$$, and therefore number of units in the $$L$$<sup>th</sup> layer, is determined by the shape of the ground truth ($$y$$).

In the case of binary classification (i.e. the target only has two classes, e.g. male or female, malignant or benign, etc., which is encoded as 0 or 1), we know for certain that the ground truth will be a 1-by-$$m$$ vector, but multi-class classification can lead to the ground truth being a higher order tensor. 

<table>
<td>
<details>
<summary>
<b>How can our target (a.k.a. ground truth, dependent variable, response, etc.) be a higher order tensor (e.g. a matrix, order-three tensor, etc.)?
</b>
</summary>
<p>

Say that our target contains nominal data. This means it won’t be proper to use a casual label encoding because we could encode the incorrect information. 
<br><br>
An example of label encoding of nominal data would be if the target contained names of North American countries and we encode “Canada” as 1, “Mexico” as 2, and “United States” as 3. 
<br><br>
This encoding implies that Mexico is thrice as weighted as Canada (because we encoded the former as 3 and the later as 1) in whatever property that the target is tracking (countryness? North Americanness? Yeah, not making much sense). The target is keeping track of the identity of countries, which just makes an ordinal representation completely unreasonable.
 <br><br>
One way to better encode nominal data is to use onehot encoding. The idea is to encode each datum as a onehot vector (a vector that contains only zeros except for one element that will be one). In the example of countries above, we would have “Canada” as the onehot vector $ \left[\begin{matrix}1\\0\\0\\\end{matrix}\right] $ and “Mexico” as $ \left[\begin{matrix}0\\1\\0\\\end{matrix}\right] $ and “United States” as $ \left[\begin{matrix}0\\0\\1\\\end{matrix}\right] $. All three vectors have same magnitude while still being different. This means our ground truth will no longer be a 1-by-$ m $, but a $ c $-by-$ m $ matrix, where $ c $ is the number of classes in the target.
<br><br>
For example, the ground truth could look something like this:
$$
\mathbf{Y}=\left[\begin{matrix}``Canada"&``Mexico"&``United\ States"&``Mexico"&``Canada"\\\end{matrix}\right]
$$

$$
\mathbf{Y}=\left[\begin{matrix}\begin{matrix}1&0&0\\\end{matrix}&0&1\\\begin{matrix}0&1&0\\\end{matrix}&1&0\\\begin{matrix}0&0&1\\\end{matrix}&0&0\\\end{matrix}\right]
$$

But with binary classification, the ground truth will always be a 1-by-$ m $ vector, $ \vec{y} $.

</p>
</details>
</td>
</table>

In the analysis covered in this writeup, we will restrict our network to just binary classification or regression. The reason is to preclude the need for a softmax output layer (an output layer that’s using the softmax activation function), which is the preferred kind of output layer for dealing with multiclass classification problems. The reason we are excluding the softmax output layer is because it will add another stratum of complexity to the backprop calculations, and that’s best kept for a separate article as multiclass classification is a topic in its own right.

However, given the premise we decided to work with, we should've denoted the activation of the last layer as $$\vec{a}^{(L)}$$ because we know it can only be a 1-by-$$m$$ vector or a scalar (actually a 1-by-1 vector) if a batch of only 1 datapoint. But we will eschew all that and continue to denote it as $$\mathbf{A}^{(L)}$$. This is purely for the sake of aesthetics and uniformity with the notations for the hidden layers, because the hidden layers are rightly denoted as $$\mathbf{A}^{(l)}$$ as they are matrices (unless we limit each hidden layer to only one unit, which results in a vector).