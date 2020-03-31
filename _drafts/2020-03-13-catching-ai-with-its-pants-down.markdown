---
layout: post
title:  "Catching AI with its pants down"
logline: "Detailed writeup exploring the inner workings of neural nets and how to biuld a standard feedforward neural net from scratch."
date:   "2020-03-13"
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


### **Gradient Descent Algorithm**
We have a loss function that is a function of the weights and biases, and we need a way to find the set of weights and biases that minimizes the loss. This is a clearcut optimization problem.

There are many ways to solve this optimization problem, but we will go with the one that scales excellently with deep neural networks, since that is the eventual goal of this writeup. And that brings us to the gradient descent algorithm.

We will illustrate how it works using a simple scenario where we have a dataset made of one feature and one target, and we want to use the mean square error as cost function. We specify a linear activation function ($$a=f(a)$$) for the neuron. Then the equation for our neuron will be:

$$
a=f\left(z\right)=w_1\ \cdot x_1+w_0
$$

Our cost function will be:

$$
J=\frac{1}{m}\cdot\sum_{j=0}^{m}{({y}_j-a_j)}^2=\frac{1}{m}\cdot\sum_{j=0}^{m}{(y_j-\ w_{1,j}\ \cdot x_{1,j}+w_{0,j})}^2
$$

Let’s further simplify our scenario by assuming we will only run computations for only one datapoint at a time.

$$
J={(Y_j-\ w_{1,j}\ \cdot x_{1,j}+w_{0,j})}^2
$$

If we hold $$Y_j$$ and $$x_{1,j}$$ constant, which is logical since they come directly from data, we observe that our cost is a function of just the parameters $$w_0$$ and $$w_1$$. And we can easily plot the curve.

{% include image.html url="/assets/images/artificial_neuron/error_vs_parameters.png" description="Plot of cost against two parameters." %}

From the plot we can easily see what values we can set $$w_0$$ and $$w_1$$ to in order to produce the most minimal cost. Any picks for $$w_0$$ and $$w_1$$ from the bottom of the valley will give a minimal cost.

The gradient descent algorithm formalizes the idea we just followed. It pretty much says: Start somewhere on the cost function (in this case, the plotted surface) and only take steps in the direction of negative gradient (i.e. direction of descent). Once you hit a minimum, any step you take will always turn out to be in the direction of ascent and therefore the iteration will no longer improve the minimization of the cost.

In mathematical terms, it is this:

$$
w_{new}=w_{old}-\gamma\frac{\partial J}{\partial w_{old}}
$$

$$
b_{new}=b_{old}-\gamma\frac{\partial J}{\partial b_{old}}
$$

{% include indent_paragraph.html content=
"Where $ \gamma $ is the step size (a.k.a. learning rate)."
%}

The above equations are used in updating the parameters at the end of each round or iteration of training. The equations are applied to each of the parameters in the model (the artificial neuron). For instance, for our toy dataset, there would be two weights, one for each feature (input variable) of the dataset, and one bias for the bias node. All three parameters will be updated using the equations. That marks the end of one round or iteration of training.

Stochastic gradient descent means that randomization is introduced during the selection of the batch of datapoints to be used in the calculations of the gradient descent. Some people will distinguish further by defining mini-batch stochastic gradient descent as when a batch of datapoints is randomly selected from the dataset and used, while stochastic gradient descent refers to just using a single randomly selected datapoint for each entire round of computations.

If we had more than two parameters, or a non-linear activation function, or some other property that makes our neuron more complicated, using a plot to find the parameters that minimize the error becomes just impractical. We must use the mathematical formulation.

What remains to be answered is how we can efficiently compute $$\frac{\partial J}{\partial w_{old}}$$ and $$\frac{\partial J}{\partial b_{old}}$$.

### **Chain rule for cost gradient**
Heads up: For this part, which is the real meat of the training process, I advised that you bring out a pen and some paper to work along, especially if this is your first time working with Jacobians.

Let’s focus on just $$\frac{\partial J}{\partial w}$$ for now. To compute the cost gradient $$\frac{\partial J}{\partial w}$$ we simply use the chain rule.

$$\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w}$$

The above chain rule equation is a core step of backpropagation (abbrev. backprop) for a neural network. This is why you may have heard of backprop being describe as just “good 'ol chain rule”, but that is a disservice to the beauty of this mathematical process, because there is one more important aspect to it, which is the reusing of values already computed or available during forward propagation, saving you some computation (while using up more computer memory).

The gradient $$\frac{\partial J}{\partial a}$$ (can also be called a Jacobian, because it is) depends on the choice of the cost function because we can’t do anything if we haven’t picked what function to use for $$J$$. Also, $$\frac{\partial a}{\partial z}$$ depends on the choice of activation function.

But for $$\frac{\partial z}{\partial w}$$, we know that preactivation (z), at least for one neuron, will always be a simple linear combination of the parameters and the input data:

$$
\boldsymbol{z}=\boldsymbol{wX}+\boldsymbol{b}$$

This is also always true in standard feedforward neural networks (a.k.a. multilayer perceptron), but not so for every flavour of neural networks (e.g. convolutional neural networks have a convolution operation instead of a multiplication between $$\boldsymbol{w}$$ and $$\boldsymbol{X}$$).

Before we move any further, it’s important you understand what Jacobians are. In a nutshell, the Jacobian of a vector function, which is what we are working with here, is a matrix that contains all of the function’s first order partial derivatives. It is the way to properly characterize the partial derivatives of a vector function. If you were not already familiar with Jacobians or still unclear of what it is, I found [this video](https://www.youtube.com/watch?v=bohL918kXQk) that should help (or just search for “Jacobian matrix” on YouTube and you’ll see many great introductory videos).


Our Jacobians in matrix representation are as follows:

$$
\frac{\partial J}{\partial w}=\left[\begin{matrix}\frac{\partial J}{\partial w_1}&\frac{\partial J}{\partial w_2}&\cdots&\frac{\partial J}{\partial w_n}\\\end{matrix}\right]
$$

$$
\frac{\partial J}{\partial a}=\left[\begin{matrix}\frac{\partial J}{\partial a_1}&\frac{\partial J}{\partial a_2}&\cdots&\frac{\partial J}{\partial a_m}\\\end{matrix}\right]
$$

$$
\frac{\partial a}{\partial z}=\left[\begin{matrix}\frac{\partial a_1}{\partial z_1}&\frac{\partial a_1}{\partial z_2}&\cdots&\frac{\partial a_1}{\partial z_m}\\\frac{\partial a_2}{\partial z_1}&\frac{\partial a_2}{\partial z_2}&\cdots&\frac{\partial a_2}{\partial z_m}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial a_m}{\partial z_1}&\frac{\partial a_m}{\partial z_2}&\cdots&\frac{\partial a_m}{\partial z_m}\\\end{matrix}\right]
$$

$$
\frac{\partial z}{\partial w}=\left[\begin{matrix}\frac{\partial z_1}{\partial w_1}&\frac{\partial z_1}{\partial w_2}&\cdots&\frac{\partial z_1}{\partial w_n}\\\frac{\partial z_2}{\partial w_1}&\frac{\partial z_2}{\partial w_2}&\cdots&\frac{\partial z_2}{\partial w_n}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial z_m}{\partial w_1}&\frac{\partial z_m}{\partial w_2}&\cdots&\frac{\partial z_m}{\partial w_n}\\\end{matrix}\right]
$$

{% include indent_paragraph.html content=
"Where their shapes are: $ \frac{\partial J}{\partial w} $ is $ 1 $-by-$ n $, $ \frac{\partial J}{\partial a} $ is $ 1 $-by-$ m $, $ \frac{\partial a}{\partial z} $ is $ m $-by-$ m $, and $ \frac{\partial z}{\partial w} $ is $ m $-by-$ n $."
%}

The shapes show us that matrix multiplication present in the chain rule expansion is valid.

From the above equation for $$z$$, we can immediately compute the Jacobian $$\frac{\partial z}{\partial w}$$.

We can observe that the Jacobian $$\frac{\partial z}{\partial w}$$ is an $$m$$-by-$$n$$ matrix. But at this stage, our Jacobian hasn’t given us anything useful because we still need the solution for each element of the matrix. 

We’ll solve an arbitrary element of the Jacobian and extend the pattern to the rest. Let’s begin.

We pick an element $$\frac{\partial z_j}{\partial w_i}$$ from the matrix, and immediately we observe that we have already encountered the generalized elements $$z_j$$ and $$w_i$$ in the following equation:

$$
z_j=w_1\ \cdot x_{1,j}+w_2\ \cdot x_{2,j}+\ldots+w_n\ \cdot x_{n,j}+w_0=\sum_{i=0}^{n}{w_i\ \cdot x_{i,j}}
$$

Therefore:

$$
\frac{\partial z_j}{\partial w_i}=\frac{\partial\left(\sum_{i=0}^{n}{w_i\ \cdot x_{i,j}}\right)}{\partial w_i}
$$

The above is a partial derivative w.r.t. $$w_i$$, so we temporarily consider $$x_{i,j}$$ to be a constant.

$$
\frac{\partial z_j}{\partial w_i}=\frac{\partial\left(\sum_{i=0}^{n}{w_i\ \cdot x_{i,j}}\right)}{\partial w_i}=x_{i,j}
$$

(If it’s unclear how the above worked out, expand out the summation and do the derivatives term by term, and keep in mind that $$x_{i,j}$$ is considered to be constant, because this is a partial differentiation).

We substitute the result back into the Jacobian:

$$
\frac{\partial z}{\partial w}=\left[\begin{matrix}x_{1,1}&x_{2,1}&\cdots&x_{n,1}\\x_{1,2}&x_{2,2}&\cdots&x_{n,2}\\\vdots&\vdots&\ddots&\vdots\\x_{1,m}&x_{2,m}&\cdots&x_{n,m}\\\end{matrix}\right]
$$

Recall that we originally defined $$\boldsymbol{X}$$) as:

$$
X=\left[\begin{matrix}x_{1,1}&x_{1,2}&\cdots&x_{1,m}\\x_{2,1}&x_{2,2}&\cdots&x_{2,m}\\\vdots&\vdots&\ddots&\vdots\\x_{n,1}&x_{n,2}&\cdots&x_{n,m}\\\end{matrix}\right]
$$

Therefore, we observe that $$\frac{\partial z}{\partial w}$$ is exactly the transpose of our original definition of X:

$$
\frac{\partial z}{\partial w}=X^T
$$

One Jacobian is down. Two more to go.

The Jacobian $$\frac{\partial a}{\partial z}$$ depends on the choice of activation function, since it is obviously the gradient of the activation w.r.t. to preactivation (i.e. the derivative of the activation function). We cannot characterize it until we fully characterize the equation for $$a$$.

Let’s go with the logistic activation function:

$$
a=\frac{1}{1+e^{-z}}
$$

$$
\frac{\partial a}{\partial z}=\left[\begin{matrix}\frac{\partial a_1}{\partial z_1}&\frac{\partial a_1}{\partial z_2}&\cdots&\frac{\partial a_1}{\partial z_m}\\\frac{\partial a_2}{\partial z_1}&\frac{\partial a_2}{\partial z_2}&\cdots&\frac{\partial a_2}{\partial z_m}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial a_m}{\partial z_1}&\frac{\partial a_m}{\partial z_2}&\cdots&\frac{\partial a_m}{\partial z_m}\\\end{matrix}\right]
$$

We follow the same steps as done with the first Jacobian.

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{1}{1+e^{-z_k}}\right)}{\partial z_j}
$$

The reason for $$k$$ is that we need a subscript that conveys the idea that $$a$$ and $$z$$ in $$\frac{\partial a}{\partial z}$$ may not always have matching subscripts That is, we are considering all the elements of the Jacobian and not just the ones along the diagonal, which are the only elements that will have matching subscripts. However, both subscripts, $$j$$ and $$k$$, are tracking the same quantity, which is datapoints.

Let’s rearrange the activation function a little by multiplying both numerator and denominator by $$e^z_k$$.

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{1}{1+e^{-z_k}}\cdot\frac{e_k^z}{e_k^z}\right)}{\partial z_j}=\frac{\partial\left(\frac{e_k^z}{e_k^z+1}\right)}{\partial z_j}
$$

The reason for this is to make the use of the [quotient rule of differentiation](https://en.wikipedia.org/wiki/Quotient_rule) for solving the derivative easier to work with.

We have to consider two possible cases. One is where $$k$$ and $$j$$ are equal, e.g. $$\frac{\partial a_2}{\partial z_2}$$, and the other is when they are not, e.g. $$\frac{\partial a_1}{\partial z_2}$$.

For $$k\neq j$$:

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{e^{z_k}}{e^{z_k}+1}\right)}{\partial z_j}=0
$$

If it’s unclear how the above worked out, then notice that when $$k\neq j$$, $$z_k$$ is temporarily a constant because we are differentiating w.r.t. $$z_j$$.

For $$k=j$$:

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{e^{z_k}}{e^{z_k}+1}\right)}{\partial z_k}
$$

We apply the quotient rule of differentiation:

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{e^{z_k}}{e^{z_{k}}+1}\right)}{\partial z_k}=\frac{e^{z_k}\left(e^{z_k}+1\right)-\left(e^{z_k}\right)^2}{\left(e^{z_k}+1\right)^2}
$$

We can sort of see the original activation function somewhere in there, so we rearrange the terms and see if we can get something more compact:

$$
\frac{\partial a_k}{\partial z_j}=\frac{e^{z_k}\cdot\left(e^{z_k}+1\right)-\left(e^{z_k}\right)^2}{\left(e^{z_k}+1\right)^2}=\frac{\left(e^{z_k}\right)^2+e^{z_k}-\left(e^{z_k}\right)^2}{\left(e^{z_k}+1\right)^2}= \color{magenta}{\frac{e^{z_k}}{e^{z_k}+1}} \cdot\left(\frac{1}{e^{z_k}+1}\right)
$$

Now we clearly see the original activation function in there (in <font color="magenta">magenta</font>). But the other term also looks very similar, so we rework it a little more:

$$
\frac{\partial a_k}{\partial z_j}=\frac{e^{z_k}}{e^{z_k}+1}\cdot\left(\frac{1}{e^{z_k}+1}\right)=\color{magenta}{\frac{e^{z_k}}{e^{z_k}+1}}\cdot\left(1-\color{magenta}{\frac{e^{z_k}}{e^{z_k}+1}}\right)
$$

We can now simply substitute it in the activation (while recalling that $$k\ =\ j$$):

$$\frac{\partial a_k}{\partial z_j}=a_k\cdot\left(1-a_k\right)=a_j\cdot\left(1-a_j\right)$$

Therefore, our Jacobian becomes:

$$
\frac{\partial a}{\partial z}=\left[\begin{matrix}a_1\cdot\left(1-a_1\right)&0&\cdots&0\\0&a_2\cdot\left(1-a_2\right)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&a_m\cdot\left(1-a_m\right)\\\end{matrix}\right]
$$

It’s an $$m$$-by-$$m$$ diagonal matrix.

Two Jacobians are down and one more to go.

However, I will leave the details for the last Jacobian $$\frac{\partial J}{\partial a}$$ as an exercise for you (it’s not more challenging than the other two). Here's the setup for it.

The cost gradient $$\frac{\partial J}{\partial a}$$ depends on the choice of the cost function since it is obviously the gradient of the cost w.r.t. activation. Since we are using a logistic activation function, we will go ahead and use the logistic loss function (a.k.a. cross entropy loss or negative log-likelihoods):

$$J=-\frac{1}{m}\cdot\sum_{j}^{m}{y_j\cdot l o g{(a}_j)+(1-y_j)\cdot\log({1-a}_j)}$$

The result for $$\frac{\partial J}{\partial \boldsymbol{a}}$$ is:

$$
\frac{\partial J}{\partial\boldsymbol{a}}=-\frac{1}{m}\cdot\left(\frac{ \boldsymbol{y}}{\boldsymbol{a}}-\frac{1-\boldsymbol{y}}{1-\boldsymbol{a}}\right)
$$

Note that all the arithmetic operations in the above are all elementwise. The resulting cost gradient is a vector that has same shape as $$a$$ and $$y$$, which is $$1$$-by-$$m$$.

Now we recombine everything. Therefore, the equation for computing the cost gradient for an artificial neuron that uses a logistic activation function and a cross entropy loss is:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{a}}\frac{\partial \boldsymbol{a}}{\partial z}\frac{\partial z}{\partial \boldsymbol{w}}=-\frac{1}{m}\cdot\left(\frac{\boldsymbol{y}}{\boldsymbol{a}}-\frac{1-\boldsymbol{y}}{1-\boldsymbol{a}}\right)\frac{\partial a}{\partial z}X^T
$$

We choose to combine the first two gradients into $$\frac{\partial J}{\partial \boldsymbol{z}}$$ such that $$\frac{\partial J}{\partial w}$$ is:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T
$$

The gradient $$\frac{\partial J}{\partial\boldsymbol{z}}$$ came from this:

$$
\frac{\partial J}{\partial\boldsymbol{z}}=\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}
$$

We already have everything for  $$\frac{\partial J}{\partial \boldsymbol{z}}$$:

$$
\frac{\partial J}{\partial \boldsymbol{z}}=\color{brown}{\frac{\partial J}{\partial \boldsymbol{a}}}\color{blue}{\frac{\partial \boldsymbol{a}}{\partial z}}=\color{brown}{-\frac{1}{m}\cdot\left(\frac{ \boldsymbol{y}}{ \boldsymbol{a}}-\frac{1- \boldsymbol{y}}{1- \boldsymbol{a}}\right) }\color{blue}{\left[\begin{matrix}a_1\cdot\left(1-a_1\right)&0&\cdots&0\\0&a_2\cdot\left(1-a_2\right)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&a_m\cdot\left(1-a_m\right)\\\end{matrix}\right]}
$$

$$
\frac{\partial J}{\partial w}=\frac{\partial J}{\partial\boldsymbol{z}}X^T=\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}X^T
$$

{% include indent_paragraph.html content=
"
Where $ \frac{\partial J}{\partial w} $ is $ 1 $-by-$ n $, $ \frac{\partial J}{\partial z} $ is $ 1 $-by-$ m $, $ \frac{\partial J}{\partial\boldsymbol{a}} $ is a $ 1 $-by-$ m $ vector,  $ \frac{\partial\boldsymbol{a}}{\partial z} $ is an $ m $-by-$ m $ matrix. Note that division between vectors or matrices, e.g. $ \frac{\boldsymbol{y}}{\boldsymbol{a}} $, are always elementwise."
%}

Notice that everything needed for computing the vital cost gradient $$\frac{\partial J}{\partial w}$$ has either already been computed during forward propagation or is from the data. We are simply reusing values already computed prior.

The above equation can now be easily implemented in code in a vectorized fashion. Implementing the code for computing the gradient $$\frac{\partial\boldsymbol{a}}{\partial \boldsymbol{z}}$$ in a vectorized fashion is a little tricky. To compute it, we first compute its diagonal as a row vector:

$$
diagonal\ vector\ of\ \frac{\partial\boldsymbol{a}}{\partial z}=(\boldsymbol{a}\odot\left(1-\boldsymbol{a}\right))
$$

$$
=\left[\begin{matrix}a_1\cdot(1-a_1\ )&a_2\cdot(1-a_2\ )&\cdots&a_m\cdot(1-a_m\ )\\\end{matrix}\right]
$$

{% include indent_paragraph.html content=
"Where $ \boldsymbol{a} $ is the $ 1 $-by-$ m $ vector that contains the activations. The symbol $ \odot $ represents elementwise multiplication (a.k.a. Hadamard product). 
<br><br>
The $ diagonal\ vector\ of\ \frac{\partial\boldsymbol{a}}{\partial \boldsymbol{z}} $ is the $ 1 $-by-$ m $ vector that you will obtain if you pulled out the diagonal of the matrix $ \frac{\partial\boldsymbol{a}}{\partial \boldsymbol{z}} $ and put it into a row vector."
%}


We also observe that the $$diagonal\ vector\ of\frac{\partial\boldsymbol{a}}{\partial z}$$ (the vector that you get if you pulled out the diagonal of the matrix $$\frac{\partial\boldsymbol{a}}{\partial z}$$ and put it into a row vector) is simply the elementwise derivative of the vector $$\boldsymbol{z}$$:

$$
diagonal\ vector\ of\frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}}=\left[\begin{matrix}a_1\cdot\left(1-a_1\ \right)&a_2\cdot\left(1-a_2\ \right)&\cdots&a_m\cdot\left(1-a_m\ \right)\\\end{matrix}\right]
$$

$$
=\ \left[\begin{matrix}f'(z_1)&f'(z_2)&\cdots&f'(z_m)\\\end{matrix}\right]=f'(\boldsymbol{z})
$$

So, computing the $$diagonal\ vector\ of\frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}}$$ is simply same as computing $$f'(\boldsymbol{z})$$, and this applies to any activation function $$f$$ and its derivative $$f'$$. And this is easily implemented in code.



<table>
<td>
<details>
<summary>
<b>Why the $ diagonal\ vector\ of\frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}} $ is always equal to $ f'(\boldsymbol{z}) $ for any activation function:
</b>
</summary>
<p>
The reason why the expression, $ diagonal\ vector\ of\frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}}=f\prime(\boldsymbol{z}) $, is valid for the logistic activation function is precisely because of this result (already shown before):

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{e^{\boldsymbol{z}_\boldsymbol{k}}}{e^{\boldsymbol{z}_\boldsymbol{k}}+1}\right)}{\partial z_j}=0
$$

{% include indent_paragraph.html content="
For $ k\neq j $. Where both $ j $ and $ k $ track the same quantity, which is datapoints."
%}

The above equation tell us that the only time an element of the matrix $ \frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}} $ has a chance of being non-zero is when $ k=j $, which is the diagonal.
<br><br>
The great thing is that the above equation also holds true for any activation function because the reason it results in zero for the logistic activation function has nothing to do with the activation function but simply because under the condition of $ k\neq j $, the following is also true: $ z_k\neq z_j $.
<br><br>
Therefore, in general the following expression will hold true for any activation function $ f $:

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial f(z_k)}{\partial z_j}=0
$$

Which also means for any activation function $ f $, the following is also true:

$$
diagonal\ vector\ of\frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}}=f\prime(\boldsymbol{z})
$$
</p>
</details>
</td>
</table>






Once we’ve computed the $$diagonal\ vector\ of\ \frac{\partial\boldsymbol{a}}{\partial z}$$, which is a $$1$$-by-$$m$$ vector, we will implement some code that can inflate the diagonal matrix $$\frac{\partial\boldsymbol{a}}{\partial z}$$ by padding it with zeros. If coding in Python and using the NumPy library for our vectorized computations, then the method [`numpy.diagflat`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.diagflat.html) does exactly that.

One good news is that we can take the equation $$\frac{\partial J}{\partial w}=\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}X^T$$ to an alternative form that would allow us to skip the step of inflating the $$diagonal\ vector\ of\ \frac{\partial\boldsymbol{a}}{\partial z}$$ and therefore saves us a little processing time.

There is a well-known relationship between the multiplication of a vector with a diagonal matrix, and elementwise multiplication (a.k.a. Hadamard product), which is denoted as $$\odot$$. The relationship plays out like this. 

Say we have a row vector $$v$$ and a diagonal matrix $$D$$, and when we flatten the $$D$$ into a row vector $$d$$ (that is, we pull out the diagonal from $$D$$ and put it into a row vector), whose elements is just the diagonal of $$D$$, then we can write:

$$
\color{brown}{v}\color{blue}{D}=\color{brown}{v} \odot \color{blue}{d}
$$

(Test out the above for yourself with small vectors and matrices and see if the two sides indeed equate to one another).

We apply this relationship to our gradients and get:

$$
\frac{\partial J}{\partial \boldsymbol{z}}=\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}=\frac{\partial J}{\partial \boldsymbol{a}}\odot\left(diagonal\ vector\ of\ \frac{\partial\boldsymbol{a}}{\partial z}\right)
$$

In fact, we can casually equate $$\frac{\partial\boldsymbol{a}}{\partial z}$$ to $$f'(\boldsymbol{z})$$, which is same as its diagonal vector. The math works out in a very nice way in that it gives the impression that we are extracting only useful information from the matrix (which is the diagonal of the matrix). 

Therefore, we end up perfoming the following assignment operation:

$$
\frac{\partial\boldsymbol{a}}{\partial z}:=f'(\boldsymbol{z})=(\boldsymbol{a}\odot\left(1-\boldsymbol{a}\right))
$$

{% include indent_paragraph.html content=
"Note that the symbol := means that this is an assignment statement, not an equation. That is, we are setting the term on the LHS to represent the terms on the RHS."
%}

Therefore, our final equation for computing the cost gradient $$\frac{\partial J}{\partial w}$$ can be written as:

$$
\frac{\partial J}{\partial w}=\frac{\partial J}{\partial\boldsymbol{z}}\frac{\partial z}{\partial w}=\ \frac{\partial J}{\partial\boldsymbol{z}}X^T=\frac{\partial J}{\partial\boldsymbol{a}}\odot\frac{\partial\boldsymbol{a}}{\partial z}X^T=\frac{\partial J}{\partial\boldsymbol{a}}\odot f'(\boldsymbol{z})X^T
$$

$$
=-\frac{1}{m}\bullet\left(\frac{\boldsymbol{y}}{\boldsymbol{a}}-\frac{1-\boldsymbol{y}}{1-\boldsymbol{a}}\right)\ \odot(a\odot\left(1-a\right))X^T
$$

{% include indent_paragraph.html content=
"
Where $ \frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}} $ here is just the diagonal of the actual $ \frac{\partial\boldsymbol{a}}{\partial\boldsymbol{z}} $ and has a shape of $ 1 $-by-$ m $ and is equal to $ f'(\boldsymbol{z}) $.
<br><br>
Note that we applied a property of how Hadamard product interacts with matrix multiplication: $ \left(v \odot u\right)M = v\odot uM = \left(u \odot v\right)M=u\odot vM $. Where $ v $ and $ u $ are vectors of same length, and $ M $ is a matrix for which the matrix multiplication shown are valid."
%}

Now for $$\frac{\partial J}{\partial b}$$, we can borrow a lot of what we did for $$\frac{\partial J}{\partial w}$$ here as well.

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial\boldsymbol{z}}\ \frac{\partial\boldsymbol{z}}{\partial b}=\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}\frac{\partial z}{\partial b}
$$

We know that $$\frac{\partial J}{\partial b}$$ has to be a scalar (or $$1$$-by-$$1$$ vector) because there is only one bias in the model, unlike weights, of which there are $$n$$ of them. During gradient descent, there is only one bias value to update, so if we have a vector or matrix for $$\frac{\partial J}{\partial b}$$, then we won’t know what to do with all those values in the vector or matrix.

We have to recall that the only reason that $$b$$ is a $$1$$-by-$$m$$ vector in the equations for forward propagation is because it gets stretched (broadcasted) into a $$1$$-by-$$m$$ vector to match the shape of $$z$$, so that the equations are valid. Fundamentally, it is a scalar and so is $$\frac{\partial J}{\partial b}$$.

Although the further breakdown of $$\frac{\partial J}{\partial\boldsymbol{z}}$$ into $$\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}$$ is shown above, we won’t need to use that since we already fully delineated $$\frac{\partial J}{\partial\boldsymbol{z}}$$ earlier. So, we just tackle $$\frac{\partial J}{\partial\boldsymbol{z}}\frac{\partial\boldsymbol{z}}{\partial\boldsymbol{b}}$$. 

Actually, just need $$\frac{\partial\boldsymbol{z}}{\partial b}$$ since we already have $$\frac{\partial J}{\partial\boldsymbol{z}}$$. The matrix representation of $$\frac{\partial\boldsymbol{z}}{\partial b}$$ is:

$$
\frac{\partial\boldsymbol{z}}{\partial b}=\left[\begin{matrix}\frac{\partial z_1}{\partial b}\\\frac{\partial z_2}{\partial b}\\\vdots\\\frac{\partial z_m}{\partial b}\\\end{matrix}\right]\ 
$$

Let’s work on it but keeping things in compact format:

$$
\frac{\partial\boldsymbol{z}}{\partial b}=\frac{\partial(\boldsymbol{wX}\ +\ \boldsymbol{b})}{\partial b}=\frac{\partial(\boldsymbol{wX})}{\partial b}+\frac{\partial\boldsymbol{b}}{\partial b}=0+\frac{\partial\boldsymbol{b}}{\partial b}=\frac{\partial\boldsymbol{b}}{\partial b}
$$

Let’s examine $$\frac{\partial\boldsymbol{b}}{\partial b}$$. It’s an m-by-1 vector that is equal to $$\frac{\partial\boldsymbol{z}}{\partial b}$$, which also means it has same shape as $$\frac{\partial\boldsymbol{z}}{\partial b}$$. You also observe that it has the shape of $$z^T$$. 

When you transpose a vector or matrix, you also transpose their shape, which fortunately is simply done by reversing the order of the shape, so when a 1-by-$$m$$ vector is transposed, its new shape is $$m$$-by-1. So, $$\frac{\partial\boldsymbol{b}}{\partial b}$$ looks like this:

$$
\frac{\partial\boldsymbol{b}}{\partial b}=\left[\begin{matrix}\frac{\partial b}{\partial b}\\\frac{\partial b}{\partial b}\\\vdots\\\frac{\partial b}{\partial b}\\\end{matrix}\right]=\left[\begin{matrix}1\\1\\\vdots\\1\\\end{matrix}\right]\
$$

Therefore $$\frac{\partial z}{\partial\boldsymbol{b}}$$ is a vector of all ones that has the shape $$m$$-by-$$1$$ (the shape of $$z^T$$).

$$
\frac{\partial\boldsymbol{z}}{\partial b}=\frac{\partial\boldsymbol{b}}{\partial b}=\left[\begin{matrix}\frac{\partial b}{\partial b}\\\frac{\partial b}{\partial b}\\\vdots\\\frac{\partial b}{\partial b}\\\end{matrix}\right]=\left[\begin{matrix}1\\1\\\vdots\\1\\\end{matrix}\right]
$$

Thus $$\frac{\partial J}{\partial b}$$ is fully delineated:

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial\boldsymbol{z}}\ \frac{\partial\boldsymbol{z}}{\partial b}
$$

{% include indent_paragraph.html content=
"Where $ \frac{\partial J}{\partial\boldsymbol{z}} $ is the gradient already computed in the steps for computing $ \frac{\partial J}{\partial\boldsymbol{w}} $, and $ \frac{\partial z}{\partial b} $ is an $ m $-by-$ 1 $ vector of ones (i.e. has same shape as $ z^T $)."
%}

Therefore the Jacobian $$\frac{\partial z}{\partial b}$$ is easily implemented in code by simply creating a vector of ones whose shape is same as $ z^T $. But there is another way we can recharacterize the above equation for $$\frac{\partial J}{\partial b}$$ such that we avoid creating any new vectors.

As mentioned earlier, matrix multiplication, or specifically vector-matrix multiplication, is essentially one example of tensor contraction. 

Here is a quick overview of tensor contraction.

From the perspective of tensor contraction, the vector-matrix multiplication of a row vector $$v$$ and a matrix $$M$$ to produce a row vector $$u$$ is:

$$u_q=\sum_{p}{v_p\cdot M_{p,q}}$$

{% include indent_paragraph.html content=
"Where the subscript $ p $ tracks the only non-unit axis of the vector $ v $, and the subscript $ q $ tracks second axis of the matrix $ M $."
%}


It shows exactly the elementwise version of matrix multiplication. Here is an example to illustrate the above. Say that $$v$$ and $$M$$ are:

$$
v=\ \left[\begin{matrix}1&2\\\end{matrix}\right]
$$

$$
M=\left[\begin{matrix}3&5&7\\4&6&8\\\end{matrix}\right]
$$

{% include indent_paragraph.html content=
"The vector $ v $ is $ 1 $-by-$ 2 $, and we will use the subscript $ q $ to track the non-unit axis, i.e. the second axis (the one that counts to a maximum of 2). That is: $ v_1=1 $ and $ v_2=2 $.
<br><br>
The matrix $ M $ is $ 2 $-by-$ 3 $, and we will use the subscript $ q $ to track the first axis (the one that counts to a maximum of 2) and $ p $ to track the second axis (the one that counts to a maximum of 3). That is $ M_{2,1}=4 $ and $ M_{1,3}=7 $."
%}

We know that the vector-matrix multiplication, $$vM$$, produces a vector. Let’s call it $$u$$, and it has the shape $$1$$-by-$$3$$.

$$
u=\left[\begin{matrix}u_1&u_2&u_3\\\end{matrix}\right]
$$

Using the tensor contraction format, we can fully characterize what the resulting vector $$u$$ is, by describing it elementwise:

$$
u_q=\sum_{p}{v_p\cdot M_{p,q}}
$$

For instance,

$$
u_1=v_1\cdot M_{1,1}+v_2\cdot M_{2,1}=1\cdot3+2\cdot4=11
$$

And we can do this for $$u_2$$ and $$u_3$$ (try it). In all, we have:

$$
u=\left[\begin{matrix}11&17&23\\\end{matrix}\right]
$$

To summarize, the vector multiplication $$vM$$ is a contraction along the axis tracked by subscript $$p$$.

We can use the tensor contraction format to recharacterize our solution for $ \frac{\partial J}{\partial b} $. 

In tensor contraction format, this:

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial\boldsymbol{z}}\ \frac{\partial\boldsymbol{z}}{\partial b}
$$

Can be written as this:

$$
\frac{\partial J}{\partial b}=\sum_{j=1}^{m}{\left(\frac{\partial J}{\partial\boldsymbol{z}}\right)_j\cdot\left(\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}\right)_j}
$$

And because $$\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}$$ is a vector of ones, we have:

$$
\frac{\partial J}{\partial b}=\sum_{j=1}^{m}\left(\frac{\partial J}{\partial\boldsymbol{z}}\right)_j
$$

In essence, we just summed across the second axis of $ \frac{\partial J}{\partial z} $ which reduced it to a $1$-by-$1$ vector, which we then equated to $ \frac{\partial J}{\partial b} $.

We now have all our cost gradients fully delineated.


In all, we can summarize with this:

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial\boldsymbol{z}}\ \frac{\partial\boldsymbol{z}}{\partial b}=\sum_{j=1}^{m}\left(\frac{\partial J}{\partial\boldsymbol{z}}\right)_j
$$

{% include indent_paragraph.html content=
"Where $ \frac{\partial J}{\partial\boldsymbol{z}} $ is the gradient already computed in the steps for computing $ \frac{\partial J}{\partial\boldsymbol{w}} $, and $ \frac{\partial z}{\partial b} $ is an $ m $-by-$ 1 $ vector of ones (i.e. has same shape as $ z^T $)."
%}


Although, we don't really need to see the equation for $ \frac{\partial J}{\partial w} $ in its contraction format, we will present it for the sake of it. We already know that $ \frac{\partial J}{\partial w} $ is:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T
$$

And we also already know that $ \frac{\partial J}{\partial \boldsymbol{z}} $ is a $ 1 $-by-$ m $ row vector and $ X $ is an $ n $-by-$ m $ matrix, which makes $ X^T $ an $ m $-by-$ n $ matrix. In tensor contraction format, the above equation is:

$$
\left(\frac{\partial J}{\partial\boldsymbol{w}}\right)_i=\sum_{j=1}^{m}{\left(\frac{\partial J}{\partial\boldsymbol{z}}\right)_j\cdot\left(X^T\right)_{j,i}}
$$

If you singled out a feature from your data (i.e. a column from $$X^T$$) and replaced all of its values for all datapoints with 1, the above equation will turn exactly into the tensor contraction of the equation for $$\frac{\partial J}{\partial b}$$. This is exactly in line with what the bias node represents.

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
