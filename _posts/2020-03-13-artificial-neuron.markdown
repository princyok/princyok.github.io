---
layout: post
title:  "Catching AI with its pants down"
date:   "2020-03-13"
categories: machine-learning
permalink:
comments: true
---
{% include scripts.html %}

* TOC
{:toc}

## **Prelude: Some Musings About AI**
### **Objective**
The goal of this writeup is to present modern AI, which is largely powered by deep neural networks, in a highly accessible form. I will walk you through building a deep neural network from scratch without reliance on any machine learning libraries and we will use our network to tackle real public research datasets.

To keep this very accessible, all the mathematics will be simplified to a level that a high school graduate who can code (and did okay in math) should be able to follow. Together we will strip the mighty, massively hyped AI of its cloths.

The original plan was to explain everything in one giant article, but that quickly prove unwieldy. So, I decided to break things up into two articles. This first article covers the prelude (basically some casual ramblings about AI) and part 1 (focuses on building an artificial neuron from scratch), and the sequel article (work in progress) will go over network of artificial neurons (a.k.a. neural networks). However, the codes for both articles have been made available.

I tried to make this first writeup very detailed, simple and granular, such that by the end of the series of planned writeups, you hopefully should have enough knowledge to investigate and code more advanced architectures from scratch if you chose to do so.

### **Motivation**
My feeling is if you want to understand a really complicated device like a brain, you should build one. I mean, you can look at cars, and you could think you could understand cars. When you try to build a car, you suddenly discover then there's this stuff that has to go under the hood, otherwise it doesn't work.

The entirety of the above paragraph is one of my favourite quotes by Geoffrey Hinton, one of the three Godfathers of Deep Learning. I don’t think we need any further motivation for why we should peek under the hood to see precisely what’s really going on in a deep learning AI system.

Tearing apart whatever is under the hood has been my canon for my machine learning journey, so it was natural that I would build a deep neural network from scratch especially after I couldn’t find any such complete implementation online (as of late 2018). After I was done, some of my colleagues thought it would be good if I put together some explanation of what I did, and so the idea for this writeup was born. After dragging my feet forever, here it is.

### **Artificial General Intelligence: The Holy Grail of AI**
Artificial intelligence (AI) is the intelligence, or the impression thereof, exhibited by things made by we humans. The kind of intelligence we have is natural intelligence. A lot of things can fall under the umbrella of AI because the definition is vague. Everything from the computer player of Chess Titans on Windows 7 to Tesla’s autopilot is called AI.

Artificial general intelligence (AGI) is the machine intelligence that can handle anything a human can. You can think of the T-800 from The Terminator or Sonny from I, Robot (although in my opinion, the movie’s view of AI, at least with regards to Sonny, aligns more with symbolic, rule-based AI instead of machine learning). Such AI system is also referred to as strong AI.

{% include image.html url="/assets/images/artificial_neuron/t800_terminator.png" description="The T-800 strong AI" %}

AGI would be able to solve problems that were not explicitly specified in its design phase.
There is no AGI system in existence today, nor is there any research group that is known to be anywhere close to deploying one. In fact, there is not even a semblance of consensus on when AGI could become reality.

Tech author Martin Ford, for his 2018 book Architects of Intelligence, surveyed 23 leading AI figures about when there would be a 50 percent chance of AGI being built [ref1;ref2]. Those surveyed included DeepMind CEO Demis Hassabis, Head of Google AI Jeff Dean, and Geoffrey Hinton (one of the three Godfathers of Deep Learning).

Of the 23 surveyed, 16 answered anonymously, and 2 answered with their names. The most immediate estimate of 2029 came from Google director of engineering Ray Kurzweil and the most distant estimate of 2200 came from Rod Brooks (the former director of MIT’s AI lab and co-founder of iRobot). The average estimate was 2099.

There are many other surveys out there that give results in the 2030s and 2040s. I feel this is because people have a tendency to want the technologies that they are hopeful about to become reality in their lifetimes, so they tend to guess 20 to 30 years from the present, because that’s long enough time for a lot of progress to be made in any field and short enough to fit within their lifetime.

For instance, I too get that gut feeling that space propulsions that can reach low-end relativistic speeds should be just 20 to 40 years away; how else will the Breakthrough Starshot (founded by  Zuckerberg, Milner and the late Hawking) get a spacecraft to Proxima Centauri b. Same for fault-tolerant quantum computers, fusion power with gain factor greater than 1, etc. They are all just 20 to 40 years away, because these are all things I really want to see happen.

Also, it seems that [AI entrepreneurs tend to be much more optimistic about how close we are to AGI than AI researchers](https://blog.aimultiple.com/artificial-general-intelligence-singularity-timing/) are. Someone should do a bigger survey for that, ha!

### **Artificial Narrow Intelligence**
The type of AI we interact with today and hear of nonstop in the media is artificial narrow intelligence (ANI), also known as weak AI. It differs from AGI in that the AI is designed to deal with a specific task or a specific group of closely related tasks. Some popular examples are AlphaGo, Google Assistant, Alexa, etc.

A lot of the hype that has sprung up around ANI in the last decade was driven by the progress made with applying deep neural networks (a.k.a. deep learning) to supervised learning tasks (we will talk more about these below) and more recently to reinforcement learning tasks.

A supervised learning task is one were the mathematical model (what we would call the AI if we’re still doing buzzspeak) is trained to associate inputs with their correct outputs, so that it can later produce a correct output when fed an input it never saw during training. An example is when Google Lens recognizes the kind of shoe you are pointing the camera at, or when IBM’s Watson transcribes your vocal speech to text. Google Lens can recognize objects in images because the neural network powering it has been trained with images where the objects in them have been correctly labelled, so that when it later sees a new image it has never seen before, it can still recognize patterns that it already learned during training.

In reinforcement learning, you have an agent that tries to maximize future cumulative reward by exploring and exploiting the environment. That’s what AlphaGo is in a nutshell.

The important point is that deep neural networks have been a key transformative force in the development of powerful ANI solutions in recent times.


## **Part 1: Building an Artificial Neuron From Scratch**
### **Machine learning Overview**
The rise of the deep learning hype has been a huge boon for its parent field of machine learning. Machine learning is simply the study of building computers systems that can “learn” from examples (i.e. data). The reason for the quotes around “learn” is that the term is just a machine learning lingo for [mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) (and we will talk more about this later). We will also use the term “training” a lot, and it also refers to the same mathematical optimization.

{% include image.html url="/assets/images/artificial_neuron/training_vs_test_cat_dog_illustration.png" description="Machine learning is about how to make a computer learn the associations presented in the training set such that it can correctly label the images in the test set correctly. This is specifically supervised learning, a category of machine learning where the computer program is provided with correctly labelled examples to learn from. This is a task that is trivial for humans, but was practically impossible for computer programs to consistently perform well at it until convolutional neural networks came along. This was because it is extremely laborious to manually write programs to identify all the patterns needed to identify the primary object in the image)." %}

In machine learning, you have a model that takes in data and spits out something relevant to that data. For the task of labelling images of cats and dogs (see image above), a model will receive images as input data and then it will output the correct labels for those images.

{% include image.html 
url="/assets/images/artificial_neuron/image_to_numbers.png"
description=
"A digital image is just a collection of pixels, and each pixel is simply a box shaded with one color. For a greyscale image like above, there is only one color with intensity ranging from 0 (for pure black) to 256 (for pure white). For machine learning, we simply covert the image to a collection of numbers, e.g. an array or matrix." 
%}

Another example: With DeepMind’s AlphaGo, the program takes in the current board configuration as input data and spits out the next move to play that will maximize the chances of winning the match.

#### **Toy datasets**
Before we advance any further to artificial neurons, let’s introduce a toy dataset that will accompany subsequent discussions and be used to provide vivid illustration.

You can think of the data as being generated from an experiment where a device launches balls of various masses unto a board that can roll backward, and when it does roll back all the way to touch the sensor, that shot is recorded as high energy, otherwise it is classified as low energy.

{% include image.html url="/assets/images/artificial_neuron/toy_experiment_schematic.png" description="A schematic of the toy experiment." %}

Below is an excerpt of the dataset:

{% include image.html url="/assets/images/artificial_neuron/toy_dataset_excerpt.png" description="A few records (datapoints) from the toy dataset, showing all the features and targets (the column headings)." %}

The dataset has two features or inputs, i.e. `velocity` and `mass`, and a single output, which is `energy level` and it is binary. The last two columns are exactly the same, just that the third is the numerical version of the last and is what we actually use because we need to crunch numbers. In classification, the labels are converted to numbers for the learning process.

#### **Supervised vs. Unsupervised Learning**
The two main broad categories of machine learning are supervised learning and unsupervised learning. The main distinction between the two is that in the former the program is provided with a target variable (or labelled data in the in the context of classification, which is touched upon a few paragraphs below) and in the latter, no variable is designated as the target variable.

But don’t let the “supervision” in the name fool you, because, as of 2019, working on real-world unsupervised tasks requires more “supervision” (in the form of domain-specific tweaks) than supervised tasks (which can still benefit from domain-specific tweaks). But there is a general sense of expectation that unsupervised learning will start rivalling the success of supervised learning in terms of practical effectiveness (and also hype) within the next few years.

In supervised learning, we always have labelled data. This will have two parts. The juice of the data ($$X$$) and the variable that holds the labels ($$Y$$). $$X$$ is more formally called the input variables, independent variables, predictors, or **features**. $$Y$$ is also called the output variable, response, or dependent variable, ground truth or **target**. It’s quite useful to be able to recognize all these alternative names.

In our toy dataset, mass and velocity are our features, and energy level is our target.

Each datapoint in the dataset also goes by many names in the ML community. Names like “example”, “instance”, “record”, “observation”, etc., are all monikers for “datapoint”. I may use examples and records as alternatives to datapoint every now and then in this article, but I will mostly stick to using datapoint.


#### **Supervised learning: Regression vs Classification**

The two broad categories of supervised learning are classification and regression. In classification, the target variable has discrete values, e.g. cat and dog labels. There can’t be a value between cat and dog. It’s either a cat or a dog. Other examples would be a variable that holds labels for whether an email is spam or not spam.

In regression the target variable has continuous values, e.g. account balances. It could be -\\$50 or \\$20 or some number in between that. It could be floating point number like \\$188.5555. Really large positive or really large negative number.

#### **Reinforcement learning**
Another subset of machine learning that some consider a category of its own alongside supervised learning and unsupervised learning is reinforcement learning. It’s about building programs that take actions that affect an environment in such a way that the cumulative future reward is maximized; in other words, programs that love to win!

You may run into other sources that consider it a hybrid of both supervised and unsupervised learning; this is debatable because there is no label or correction involved in the training process, but there is a reward system that guides the learning process.

Also be careful, because reinforcement learning is not a definitive name for hybrids of the two. There are other subsets of machine learning that are truer hybrids of supervised and unsupervised learning but do not fall under reinforcement learning. For instance, generative adversarial neural networks (the family of machine learning models behind the [deepfake technology](https://www.youtube.com/watch?v=cQ54GDm1eL0)).

### **Estimators**
When you see an apple, you are able to recognize that the fruit is an apple. When the accelerator (gas pedal) of a car is pressed down, the velocity of the car changes. When you see a ticktacktoe board where the game is ongoing, a decision on what is the best next move emerges. All of these have one thing in common: there is a process that takes an input and spits out an output. The visuals of an apple is the input and the recognition of the name is the output. Pressing down of the accelerator is an input and the rate of change velocity  is the output. All of these processes can be thought of as functions. A function is the mapping of a set of inputs to a set of outputs in such a way that no two or more inputs will result in same output. Almost any process you can think of can be thought of as a function. The hard part is fully characterizing the function that underlies that process.

An estimator is a function that tries to estimate the behavior of another function whose details are not fully unknown.

An estimator is the core component of a supervised machine learning system. It goes by a billion other names including being simply called the model, approximator, hypothesis function, learner, etc. But note that some of these other names, like model and learner, can also refer to more than just the estimator.

Let’s say there is a function ($$f_{actual}$$) that takes $$X$$ as an input and spits out $$Y$$, then an estimator ($$f_{estim}$$) will take in the same $$X$$ as its input and will spit out $$\hat{Y}$$ as an output. This  $$\hat{Y}$$ will be expected to be approximately same as $$Y$$.

$$
Y=f_{actual}\left(X\right)
$$

$$
\hat{Y}=f_{estim}\left(X\right)
$$

Because we expect that there may be a difference between Y and $$\hat{Y}$$, we introduce an error term to capture that difference:

$$
\varepsilon=Y-\hat{Y}
$$

Therefore, we can then see that:

$$
Y=\hat{Y}+\varepsilon
$$

Or written as:

$$
Y=f_{actual}\left(X\right)=f_{estim}\left(X\right)+\varepsilon
$$

We notice that if we can minimize $$\varepsilon$$ down to a really small value, then we can have an estimator that behaves like the real function that we are trying to estimate:

$$
Y\approx\hat{Y}=f_{estim}\left(X\right)
$$

If you’ve heard of naïve Bayes or logistic regression, linear regression, etc., then you’ve heard of other examples of machine learning estimators. But those are not the focus of this article.

### **The brain as a function**
The computational theory of mind (CTM) says that we can interpret human cognitive processes as computational functions i.e. the human mind behaves just like a computer. Note that while this theory is considered a good model for human cognition (it was the unchallenged standard in the 1960s and 1970s and still widely subscribed to), no one has been able to show how consciousness can emerge from a system modelled on the basis of this theory, but that’s another topic for another time. For a short primer on the theory, see [this article](https://plato.stanford.edu/entries/computational-mind/) from the Stanford Encyclopedia of Philosophy.

According to CTM, if we have a mathematical model of all the computations that goes on in the brain, we should, one day, be able to replicate the capabilities of the brain with computers.

But how does the brain do what it does?

In a nutshell, the brain is made up of two main kinds of cells: glial cells and neurons (a.k.a. nerve cells). There are about 86 billion neurons and even more glial cells in the nervous system (brain, spinal cord and nerves) of an adult human [ref1]. The primary function of glial cells is to provide physical protection and other kinds of support to neurons, so we are not very interested in glial cells here. It’s the neuron we came for.

{% include image.html url="/assets/images/artificial_neuron/biological_neuron.png" description="A biological neuron is the building block of the nervous system, which includes the brain." %}

The primary function of biological neurons (to differentiate from the artificial neurons that make up an artificial neural network) is to process and transmit signals, and there are three main types, sensory neurons (concentrated in your sensory organs like eyes, ears, etc.), motor neurons (carry signals between the brain and spinal cord, and from both to the muscles), and interneurons (found only in the brain and spinal cord, and they process information).

For instance, when you grab a very hot cup, sensory neurons in the nerves of your fingers send a signal to interneurons in your spinal cord. Some interneurons pass the signal on to motor neurons in your hand, which causes you to drop the cup, while other interneurons send a signal to those in your brain, and you experience pain.

So clearly, in order to start modelling the brain, we have to first understand the neuron and try to model it.

Neurons in the brain usually work in groups known as neural circuits (or biological neural networks), where they provide some biological function. A neuron has 3 main parts: the dendrites, soma (cell body) and axon.

The dendrite of the one neuron is connected to the axon terminal of another neuron, and on and on, resulting in a network of connected neurons. This connection is known as the synapses, and there is no actual physical connection, because the neurons don’t actually touch each other. Instead, a neuron will release chemicals (neurotransmitters) that carry the electrical signal to the dendrite of the next neuron. The strength of the transmission is known as the synaptic strength. The more often signals are transmitted across a synapse, the stronger the synaptic strength becomes. This rule, commonly known as Hebb’s rule (introduced in 1949 by Donald Hebb), is colloquially stated as, “neurons that fire together wire together.” We will revisit this concept when we dive into artificial neural networks.

Neurons receive signal via their dendrites and outputs signal via their axon terminals. And each neuron can be connected to thousands of other neurons. When a neuron receives signals from other neurons, it combines all the input signals and generates a voltage, known as graded potential, on the membrane of the soma that is proportional, in size and duration, to the sum of the input signals. The graded potential gets smaller as it travels through the soma to reach the axon. If the graded potential that reaches the trigger zone (near where the axon meets the soma) is higher than a threshold value unique to the neuron, the neuron fires a huge electric signal, called the action potential, that travels down the axon and through the synapse to become the input signal for the neurons downstream.

###  **Artificial neuron**
In the 1950s, the psychologist Frank Rosenblatt introduced a very simple mathematical abstraction of the biological neuron. Using the limited knowledge of biological neurons that we had, Rosenblatt developed a model that mimicked the following behavior: signals that are received from dendrites are sent down the axon once the strength of the input signal crosses a certain threshold. The outputted signal can then serve as an input to another neuron. Rosenblatt named this mathematical model the perceptron [ref1].

Rosenblatt’s original perceptron was a simple [Heaviside function](https://en.wikipedia.org/wiki/Heaviside_step_function) that outputs zero if the input signal is equal to or less than 0 and outputs 1 if the input is greater than zero [ref1]. Therefore, zero was the threshold above which an input makes the neuron to fire. The perceptron is an example of an artificial neuron, and we will see other examples.

An artificial neuron is simply a mathematical function that serves as the elementary unit of a neural network. It is also known as a node or a unit, with the latter being very common in the machine learning publications. I may jump between these names a lot, and it’s not bad if you get used to it.

This mathematical function will have a collection of inputs, $$x_1,x_2,\ \ldots,\ x_n$$, and a single output, a, commonly known as the activation value (or post-activation value), or often without the term “value” (i.e. simply activation).

{% include image.html url="/assets/images/artificial_neuron/artificial_neuron.png" description="Diagram of an artificial neuron." %}

But what then happens inside a unit (an artificial neuron)?

The inputs that are fed into a unit are used in two key operations in order to generate the activation:

1. Summation: The inputs ($ x_i $) are multiplied with the weights ($$w_i$$) and summed together. This summation is sometimes called the preactivation value, or without the term “value”.

2. Activation function: the resulting sum (i.e. the preactivation) is passed through a mathematical function.

{% include image.html url="/assets/images/artificial_neuron/artificial_neuron_interior.png" description="Diagram of an artificial neuron showing what happens inside it." %}

The activation value can be thought of as a loose adaptation of the biological action potential, and the weights as the synaptic strength.

The algebraic representation of an artificial neuron is:

$$ 
a=f\left(z\right) 
$$

{% include indent_paragraph.html content=
"Where $ a $ is the activation, $ z $ is the preactivation, and $ f $ is the activation function (in the case of the original perceptron, this is the Heaviside function)."
%}
The preactivation $ z $ is computed as:

$$
z=w_1\ \bullet x_1+w_2\ \bullet x_2+\ldots+w_n\ \bullet x_n+w_0=\sum_{i=0}^{n}{w_i\ \bullet x_i}
$$

{% include indent_paragraph.html content=
"Where $ n $ is the number of features in our dataset." 
%}

It’s important to start putting these equations in the context of data. Using our toy dataset, the application of this equation can be demonstrated by taking any datapoint and subbing the values into the above equation. For instance, if we sub in the 0<sup>th</sup> datapoint (6.5233, 1.5484, 0), we get:

$$
z=w_1\ \bullet 6.5233+w_2\ \bullet1.5484+w_0
$$

We will keep the weights as variables for now (but that will change a few paragraphs later).
The complete algebraic representation of the original perceptron, which has the Heaviside function as its activation function, is:

$$
a =
\begin{cases}
 1 &\text{if } z > 0 \\
0 &\text{if } z \leq 0
\end{cases}
$$

If you took a moment to really look at the equation for preactivation, you will notice something is off, compared to the artificial neuron diagram. Where did $$w_0$$ come from? And what about $$x_0$$? The answer is the “bias term”. It allows our function to shift, and its presence is purely a mathematical necessity.

The variable $$w_0$$ is known as the bias, and $$x_0$$ is a constant that is always equal to one and has nothing to do with the data, unlike $$x_1$$ to $$x_n$$ that comes from the data (e.g. the pixels of the images in the case of the image classification example). That’s how $$w_0 \bullet x_0$$ reduces to just $$w_0$$.

In fact, by the time we start talking about the network of artificial neurons (also known as neural networks), we will refer to $$w_0$$ as $$b$$, and this is the letter often used in literature to represent the bias. The weights ($$w_1,\ w_2,\ \ldots,\ w_n$$) and bias ($$w_0$$ or $$b$$) collectively are known as the parameters of the artificial neuron.

<table>
<td>
<details>
<summary>
<b>
The need for the bias term:
</b>
</summary>
<p>
As already mentioned, the bias term allows our function to shift. Its presence is purely a mathematical necessity.
<br><br>
The equation for z is a linear equation:

$$
z=w_1\ \bullet x_1+w_2\ \bullet x_2+\ldots+w_n\ \bullet x_n+w_0
$$

If we limit the number of features (input variables) to only one, we get the equation of a line:

$$
z=w_1\ \bullet x_1+w_0
$$

{% include indent_paragraph.html content="Where $ w_1 $ is the slope of the line, and $ w_0 $ is the intercept." %}

Everything looks good. If we are given exactly two datapoints, we will be able to perfectly fit a line through them, and we will be able to calculate the slope and intercept of that line, thereby fully solving the equation of that line. That process of solving the equation to fit the data made of two points is “learning”. In fact, feel free to call it machine learning.
<br><br>
But what if we omitted the intercept? Well, we may never be able to perfectly fit a line through those two datapoints. Actually, we will never be able to perfectly fit a straight line through both points if it happens that the line that perfectly fits on them does not go through the origin (which is intercept of zero).
<br><br>
{% include image.html url="/assets/images/artificial_neuron/line_varying_slopes.png" description="Plot of lines of various slopes (m) all passing through the origin (c=0) and compared against two datapoints that cannot be perfectly fitted by a line whose y-intercept is 0, because a vertical shift is necessary." %}

But by having the intercept term, we can shift the line vertically.
<br><br>
In general, it goes like this: If we have a function $ f(x) $, then $ f\left(x\right)+c $ applies a vertical shift of $ c $ on the function. Whereas, $ f(x+c) $ applies a horizontal shift of $ c $. This should be enough refresher of this high school topic, and it is also the reason why we need the bias term.
<br><br>
But the presence of the bias term in our artificial neuron equation means that the true diagram should look like this:
<br><br>
{% include image.html url="/assets/images/artificial_neuron/artificial_neuron_bias_node.png" description="Diagram of an artificial neuron showwing the bias node." %}

But we don’t show the bias nodes because it is generally assumed that everyone should know that it is always there. This is important because it is common for the bias term to be completely omitted in many ML publications, because they know that you should know that it is there!"
</p>
</details>
</td>
</table>

We observe that the equation for an artificial neuron can be condensed into this:

$$a=f(x;w,b)$$

{% include indent_paragraph.html content=
"Where $ x=(x_1,\ x_2,\ldots,\ x_n) $ and $ w=(w_1,\ w_2,\ldots,w_n) $" 
%}

The equation is read as $ a $ is a function of $ x $ parameterized by $ w $ and $ b $. And in fact, we’ve just introduced vectors. One geometrical interpretation of a vector in a given space (could be 2D, 3D space, etc.) is that it is a point with a “sense” of direction, or just an arrow pointing from the origin to a point.

So effectively we have this:

$$
a=f(\boldsymbol{x};\boldsymbol{w},b)
$$

{% include indent_paragraph.html content=
"Where 
$ 
\boldsymbol{x}=\left[\begin{matrix}x_1\\x_2\\\vdots\\x_n\\\end{matrix}\right]
$ 
and 
$
\boldsymbol{w}=\left[\begin{matrix}w_1\\w_2\\\vdots\\w_n\\\end{matrix}\right]^T
$."
%}



If you are doubting, then check if this equation is correct (spoiler alert: it is correct!):

$$
\boldsymbol{wx}=\sum_{i=0}^{n}{w_i\ \bullet x_i}
$$

Note that the lack of any symbols between $$\boldsymbol{w}$$ and $$\boldsymbol{x}$$ signifies matrix-vector multiplication (or matrix-matrix multiplication), and you will see more of such throughout this article.

A useful idea for converting an equation or a system of them into a matrix or vector equation is to recognize that:

1.	Vector-vector multiplication is same as the dot product of two vectors.
2.	Dot product is simply elementwise multiplication (a.k.a. Hadamard product) followed by summation of the products.
3.	Vector-matrix multiplication directly reduces to the dot product between the row or column vectors of a matrix and a vector. This makes vector-matrix multiplication, which is a subset of matrix multiplication, one example of tensor contraction. (We will revisit this later).

So, when you see a pair of scalars getting multiplied and then the products from all such pairs are added, you should immediately suspect that such an equation can be easily substituted with a tensorized version.

A quick description of tensor: you probably already think of a vector as an array with one dimension (or axis). this makes it a first-order tensor, and a matrix is a second-order tensor as it has two axes. Similar objects with more than two axes are higher order tensors. In summary, a tensor is the generalization of vectors, matrices and higher order tensors.

The equations we've seen above are under the premise that we will be handling only one datapoint at a time. But we need to be able to handle more than one datapoint simultanously (we also need this when we start looking into neural networks because operations on matrices are easily parallelized). For this reason, we will do one more important thing to the equations we’ve seen above, which is to take them to matrix form. 

Improvement in parallelized computing is a huge reason deep learning returned to the spotlight in the last decade. Parallelization is also the reason GPUs have become a champion for machine learning, because they have thousands of cores unlike CPUs which typically have cores that number in the single digits.

Going back to our toy dataset, if we wanted to compute preactivations for the first three datapoints at once, we get these three equations (and please always keep in mind that w_0=b):

$$
z=w_1\ \bullet6.5233+w_2\ \bullet1.5484+w_0
$$

$$
z=w_1\ \bullet9.2112+w_2\ \bullet12.7141+w_0
$$

$$
z=w_1\ \bullet1.7315+w_2\ \bullet45.6200+w_0
$$

Clearly, we need a new subscript to keep track of multiple datapoints, because it’s misleading to keep equating every datapoint to just $$z$$. So, we do something like this:

$$
z_j=w_1\ \bullet x_{1,j}+w_2\ \bullet x_{2,j}+\ldots+w_n\ \bullet x_{n,j}+w_0=\sum_{i=0}^{n}{w_i\ \bullet x_{i,j}}
$$

{% include indent_paragraph.html content=
"Where the subscript $ j $ keeps track of datapoints. Or you can think of it as, $ i $ tracks the columns and $ j $ tracks rows in our toy dataset."
%}

So now we can write them as:

$$
z_1=w_1\ \bullet6.5233+w_2\ \bullet1.5484+w_0
$$

$$
z_2=w_1\ \bullet9.2112+w_2\ \bullet12.7141+w_0
$$

$$
z_3=w_1\ \bullet1.7315+w_2\ \bullet45.6200+w_0
$$

Note that the numerical subscript on $$z$$ above is not counterpart to that on $$w$$. The former tracks datapoints (rows in our toy dataset), and the latter tracks features (columns in our toy dataset).

You can already notice the system of equations. And if it had been a batch of 100 datapoints, or even the entire dataset, it starts becoming unwieldy to carry around thousands of equations. Therefore we vectorize!

We summarize the preactivations for all the datapoints in our batch:

$$
\boldsymbol{z}=all\ z_j\ in\ the\ batch=[\begin{matrix}z_1&z_2&\cdots&z_m\\\end{matrix}]
$$

{% include indent_paragraph.html content=
"Where $ m $ is the number of datapoints in our batch."
%}

What's the batch all about? In deep learning, it's very common to deal with very large datasets that may be too big to load into memory all at once, so we sample a batch from the dataset and use that to train our model. That's one iteration. We repeat the sampling for the second iteration, and continue for as many iterations as we choose to. 

Now we have all the ingredients to convert to matrix format. Our system of equation, will go from this:

$$
z_1=w_1\ \bullet x_{1,1}+w_2\ \bullet x_{2,1}+\ldots+w_n\ \bullet x_{n,1}+w_0
$$

$$
z_2=w_1\ \bullet x_{1,2}+w_2\ \bullet x_{2,2}+\ldots+w_n\ \bullet x_{n,2}+w_0
$$

$$
\vdots
$$

$$
z_m=w_1\ \bullet x_{1,m}+w_2\ \bullet x_{2,m}+\ldots+w_n\ \bullet x_{n,m}+w_0
$$

To this matrix equation:
 
$$
\boldsymbol{z}\ =\ \boldsymbol{wX}\ +\ \boldsymbol{b}
$$

I encourage you to rework the matrix equation back into the flat form if you’re unclear on how the two are the same. I promise, it will be a great refresher of math you probably saw in high school or first year of university.

The variable $$\boldsymbol{z}$$ is a $$1$$-by-$$m$$ vector, and if only one datapoint, will be a vector of only one entry (which is equivalent to a scalar).

The parameter $$\boldsymbol{w}$$ is always going to be a $$1$$-by-$$n$$ vector, regardless of the number of multiple datapoints.

$$
\boldsymbol{w}=\left[\begin{matrix}w_1\\w_2\\\vdots\\w_n\\\end{matrix}\right]^T
$$

The variable $$b$$ will be scalar if doing computation for a single datapoint, or it’ll be a $$1$$-by-$$m$$ vector if for multiple datapoints. However, in code implementation it will always be a scalar (or more correctly, a vector that has only one entry), but then it gets [broadcasted](https://docs.scipy.org/doc/numpy/user/theory.broadcasting.html#array-broadcasting-in-numpy) into a vector of the right shape during computation. (If this doesn’t make sense now, move on and return to it paragraph later after you finish the article).

If we defined $$b$$ to be a $$1$$-by-$$m$$ vector, we would have the following problems:

{% include indent_paragraph.html content=
"The neuron becomes restricted to a fixed batch size. That is, the batch size we use to train the neuron becomes a fixture of the neuron, to the point that we can’t use the neuron to carry out predictions or estimations for a different batch size."
%}

{% include indent_paragraph.html content=
"Each example in the batch will have a different corresponding value for $ b $, and this is not the case with $ w $, and it is just simply improper for the parameters to change from example to example."
%}

The variables $$\boldsymbol{X}$$ will depend on the shape of the input data that gets fed to the neuron. It could be a vector or matrix (and in neural networks they could even be higher order tensors). When multiple datapoints, it’s an $$n$$-by-$$m$$ matrix, and when a single datapoint it's an $$n$$-by-$$1$$ vector. It looks like this:

$$
X=\left[\begin{matrix}x_{1,1}&x_{1,2}&\cdots&x_{1,m}\\x_{2,1}&x_{2,2}&\cdots&x_{2,m}\\\vdots&\vdots&\ddots&\vdots\\x_{n,1}&x_{n,2}&\cdots&x_{n,m}\\\end{matrix}\right]
$$

Keep in mind that these statements about the shapes of these tensors are all for a single artificial neuron, as there are some changes when moving unto neural networks (a network of neurons).

Let’s illustrate with our toy dataset how the preactivation equation works in matrix format. Let’s say we decide that our batch size will be 3, which means we will feed our neuron 3 datapoints (3 rows of our dataset), then our $$X$$ will look like this:

$$
\boldsymbol{X}=\left[\begin{matrix}6.5233&9.2112&1.7315\\1.5484&12.7141&45.6200\\\end{matrix}\right]
$$

And the corresponding $$y$$ is this:

$$
\boldsymbol{y}=\ \left[\begin{matrix}0&1&0\\\end{matrix}\right]
$$

Let’s say we randomly initialize our weight vector to this (which is actually what is done in practice, but more like “controlled” randomization):

$$
\boldsymbol{w}=\left[\begin{matrix}w_1\\w_2\\\end{matrix}\right]^T=\left[\begin{matrix}0.5&-0.3\\\end{matrix}\right]
$$

And we set our bias to zero. Note that it will be a scalar, but broadcasted during computation to match whatever shape $$\boldsymbol{z}$$ has:

$$
b=0
$$

Then we can compute our preactivation for this batch of 3 datapoints:

$$
\boldsymbol{z}\ =\ \left[\begin{matrix}0.5&-0.3\\\end{matrix}\right]\left[\begin{matrix}6.5233&9.2112&1.7315\\1.5484&12.7141&45.6200\\\end{matrix}\right]\ +\left[\begin{matrix}0&0&0\\\end{matrix}\right]
$$

$$
z=\left[\begin{matrix}2.79713&0.79137&-12.8202\\\end{matrix}\right]
$$

Let’s assume that the kind of artificial neuron we have is the original perceptron (that is, our activation function is the Heaviside function). Recall that:

$$
a_j =
\begin{cases}
 1 &\text{if } z_j > 0 \\
0 &\text{if } z_j \leq 0
\end{cases}
$$

Now we pass $$\boldsymbol{z}$$ through a Heaviside function to obtain our activation value:

$$
\boldsymbol{a}=\left[\begin{matrix}1&1&0\\\end{matrix}\right]
$$

Remember we already have the ground truth ($$\boldsymbol{y}$$), so we can actually check and see how our (untrained) neuron did.

$$
\boldsymbol{y}=\ \left[\begin{matrix}0&1&0\\\end{matrix}\right]
$$

We can easily notice that $$\boldsymbol{a}$$, $$\boldsymbol{y}$$ and $$\boldsymbol{z}$$ will always have the same shape, which is a $$1$$-by-$$m$$ vector; and if only one datapoint, will be a vector of only one entry (which is equivalent to a scalar).

And it did okay. It got the first datapoint wrong (it predicted high energy instead of the correct label of low energy) but got the other two right. That’s 66.7% accuracy. We likely won’t be this lucky if we use more datapoints.

To improve the performance of the artificial neuron, we need to train it. That simply means that we need to find the right values for the parameters w\ and b such that when we feed our neuron any datapoint from the dataset, it will estimate the correct energy level.

This is the general idea of how the perceptron, or any other kind of artificial neuron, works. That is, we should be able to compute a set of parameters ($$w_0,\ w_1,\ w_2,\ \ldots,\ w_n$$) such that the perceptron is able to produce the correct output when given an input.

For instance, when fed the images of cats and dogs, a unit (artificial neuron) with good parameters will correctly classify them. The pixels of the image will be the input, $$x_1,x_2,\ \ldots,\ x_n$$, and the unit will do its math and output 0 or 1 (representing the two possible labels). Simple!

This is the whole point of a neural network (a.k.a. network of artificial neurons). And that process of finding a good collection of parameters for a neuron (or a network of neurons as we will see later) is what we call “learning” or “training”, which is the same thing mathematicians call mathematical optimization.

Unfortunately, the original perceptron did not fair very well in practice and failed to deliver on the high hopes heaped on it. I can assure you that it will not do too well with image classification of, say, cats and dogs. We need something more complex with some more nonlinearity.

Note that linearity is not the biggest reason Heaviside functions went out of favour. In fact, a Heaviside function is not purely linear, but instead piecewise linear. It’s also common to see lack of differentiability at zero blamed for the disfavour, but again this is cannot be the critical reason, as there are cheap tricks around this too (e.g. the same type of schemes used to get around the undifferentiability of the rectified linear function at zero, which by the way is currently the most widely used activation function in deep learning).

The main problem is that the Heaviside function jumps too rapidly, in fact instantaneously, between the two extremes of its range. That is, when traversing the domain of the Heaviside function, starting from negative to positive infinity, we will keep outputting zero (the lowest value in its range), until suddenly at the input of zero, its output snaps to 1 (the maximum value in its range) and then continues outputting that for the rest of infinity. This causes a lot of instability. When doing mathematical optimization, we typically prefer small changes to produce small changes.

### **Activation functions**
It is possible to use other kinds of functions as an activation function, and this is indeed what researchers did when the original perceptron failed to deliver. One such replacement was the sigmoid function, which resembles a smoothened Heaviside function.

It is possible to use other kinds of functions as an activation function, and this is indeed what researchers did when the original perceptron failed to deliver. One such replacement was the sigmoid function, which resembles a smoothened Heaviside function.

{% include image.html url="/assets/images/artificial_neuron/heaviside_logistic.png" description="Plots of the Heaviside and logistic activation functions." %}

Note that the term “sigmoid function” refers to a family of s-shaped functions, of which the very popular logistic function is one of them. As such, it is common to see logistic and sigmoid used interchangeably, even though they are strictly not synonyms.

The logistic function performs better than the Heaviside function. In fact, machine learning using an artificial neuron that uses the logistic activation function is one and the same as logistic regression. Ha! You’ve probably run into that one before. Don’t feel left out if you haven’t though, because you’re just about to.

This is the equation for logistic regression:

$$
\boldsymbol{\hat{y}}=\frac{1}{1+e^{-\boldsymbol{z}}}
$$

$$
\boldsymbol{z}\ =\ \boldsymbol{wX}\ +\ \boldsymbol{b}
$$

{% include indent_paragraph.html content=
"Where $ \hat{y} $ is the prediction or estimation."
%}

And this is the equation for an artificial neuron with a logistic (sigmoid) activation function:

$$
\boldsymbol{a}=\frac{1}{1+e^{-\boldsymbol{z}}}
$$

$$
\boldsymbol{z}\ =\ \boldsymbol{w}X\ +\ \boldsymbol{b}
$$

As you can see, they are one and the same!

Also note that the perceptron, along with every other kind of artificial neuron, is an estimator just like other machine learning models (linear regression, etc.).

Besides the sigmoid and Heaviside functions, there are a plethora of other functions that have found great usefulness as activation functions. **You can find a list of many other activation functions in [this Wikipedia article](https://en.wikipedia.org/w/index.php?title=Activation_function&oldid=939349877#Comparison_of_activation_functions)**. You should take note of the rectified linear function; any neuron using it is known as a rectified linear unit (ReLU). It's the most popular activation function in deep learning.

One more important mention is that the process of going from input data ($$\boldsymbol{X}$$) all the way to activation (essentially, the execution of an activation function) is called **forward pass** (or forward propagation in the context of neural networks), and this is the process we demonstrated above using the toy dataset. This distinguishes from the sequel process, known as **backward pass**, where we use the error between the activation ($$\boldsymbol{a}$$) and the ground truth ($$\boldsymbol{y}$$) to tune our parameters in such a way that the error is reduced to as low as possible.

To tie things back to our toy dataset. If we used a logistic activation function instead of a Heaviside function, and trained our neuron for 2000 iterations of training, we obtain some values for the parameters that gives us the correct result 91% of the time. (We will later go over exactly what happens during “training”).

The parameters after training are:

$$
\boldsymbol{w}=\left[\begin{matrix}0.33456&0.0206573\\\end{matrix}\right]
$$

$$
b=-2.09148
$$

So, the optimized (trained) equation for our artificial neuron is:

$$
\boldsymbol{z}\ =\ \left[\begin{matrix}0.33456&0.0206573\\\end{matrix}\right]\boldsymbol{X}\ +\left[\begin{matrix}-2.09148&-2.09148&-2.09148\\\end{matrix}\right]
$$

$$
\boldsymbol{a}=\frac{1}{1+e^{-\boldsymbol{z}}}
$$

{% include indent_paragraph.html content=
"Where $ \boldsymbol{X} $ is an $ n $-by-$ m $ matrix that contains a batch of our dataset, and $ m $ is the number of datapoints in our batch, while $ n $ is the number of features in our dataset."
%}

The above is the logistic artificial neuron that has learned the relationship hidden in our toy dataset, and you can randomly pick some datapoints in our dataset and verify the equation yourself. Roughly 9 out of 10 times, it should produce an activation that matches the ground truth ($$\boldsymbol{y}$$).

Note that the values for the parameters are not unique. A different set of values can still give us a comparable performance. We only discovered one of many possible sets of values that can give us good performance.

### **Loss function**
In the section on estimators, I mentioned that it is imperative to expect an estimator (which is what an artificial neuron is) to have some level of error in its prediction, and our objective will be to minimize this error.

We described this error as:

$$\varepsilon =y-\hat{y}=f_{actual} \left( X \right) -f_{estim} \left( X \right) $$

The above is actually one of the many ways to describe the error, and not rigorous enough. For example, it is missing an absolute value operator, else the sign of the error will change just based on how the operands of the subtraction are arranged:

$$
y-\hat{y}\neq\hat{y}-y
$$

For instance, we know the difference between the [natural numbers](https://en.wikipedia.org/wiki/Natural_number) 5 and 3 is 2, but depending on how you rearrange the subtraction between them, we could end up with -2 instead, and we don’t want that to be happening, so we apply an absolute value operation and restate the error as:

$$
\varepsilon=|y-\hat{y}|
$$

Now if we have a dataset made of more than one datapoint, we will have many errors, one for each datapoint. We need a way to aggregate all those individual errors into one big error value that we call the loss (or cost).

We achieve this by simply averaging all those errors to produce a quantity we call the mean absolute error:

$$
Mean\ Absolute\ Error=Cost=\frac{1}{m} \cdot  \sum _{j=0}^{m} \vert y_{j}-\hat{y}_{j} \vert =\frac{1}{m} \cdot  \sum _{j=0}^{m} \varepsilon _{j}
$$

{% include indent_paragraph.html content=
"Where m is the number of datapoints in the batch of data. Note that $ \hat{y}_{j} $ is same as activation $ a_j $, and it is denoted here as such to show that it serves as an estimate for the ground truth $ y_i $."
%}

The above equation happens to be just one of the many types of loss functions (a.k.a. cost function) in broad use today. They all have one thing in common: **They produce a single scalar value (the loss or cost) that captures how well our network has learned the relationship between the features and the target for a given batch of a dataset**.

<table>
<td>
<details>
<summary>
<b>Cost function vs loss function vs objective function</b>
</summary>
<p>
Some reserve the term loss function for when dealing with one datapoint and use cost function for the version that handles a batch of multiple datapoints.
<br><br>
An objective function is simply the function that gets optimized in order to solve an optimization problem. In deep learning the loss or cost function plays that role, therefore making objective function another name for loss or cost function.</p>
</details>
</td>
</table>

We will introduce two other loss functions that are very widely used.

Mean squared error loss function, which is typically used for regression tasks:

$$
Mean\ Squared\ Error:\ J=\frac{1}{m}\bullet\sum_{j}^{m}\left(y_j-a_j\right)^2
$$

You must have seen the above equation before if you’ve learned linear regression in any math course.

Logistic loss function (also known as cross entropy loss or negative log-likelihoods), which is typically used for classification tasks:

$$
Cross\ entropy\ loss:\ \ J=-\frac{1}{m}\bullet\sum_{j}^{m}{y_j\bullet \log(y_j)+(1-a_j)\bullet\log(1-a_j)}=\frac{1}{m}\bullet\sum_{j=0}^{m}\varepsilon_j
$$

Note that the logarithm in the cross entropy loss is with base $$e$$ (Euler's number). In other words, it is a natural logarithm, which is sometimes abbreviated as $$\ln$$ instead of $$\log$$.

Notice that all these loss functions have one thing in common, they are all functions of activation, which also makes them function of the parameters:

$$
Cost:\ \ J=f\left(a_j\right)=f(W,b)\
$$

For instance, cross entropy loss function for a single datapoint can be recharacterized as follows:

$$
Cross\ entropy\ loss=\ -\left(y\bullet\log{y})+(1-a)\bullet\log(1-a)\right)
$$
$$
=-\left(data+\left(1-\frac{1}{1+e^{-z}}\right)\bullet\log{\left(1-\frac{1}{1+e^{-z}}\right)}\right)\
$$
$$
=-\left(data+\left(1-\frac{1}{1+e^{\sum_{i=0}^{n}{w_i\ \bullet x_i}}}\right)\bullet\log{\left(1-\frac{1}{1+e^{-\sum_{i=0}^{n}{w_i\ \bullet x_i}}}\right)}\right)
$$
$$
=-\left(data+\left(1-\frac{1}{1+e^{\sum_{i=0}^{n}{w_i\ \bullet\ data}}}\right)\bullet\log{\left(1-\frac{1}{1+e^{-\sum_{i=0}^{n}{w_i\ \bullet\ data}}}\right)}\right)\
$$

In other words, the loss function can be described purely as a function of the parameters ($$W$$,$$b$$) and the data ($$X$$, $$y$$). And since data is known, the only unknowns on the right-hand side of the equation are the parameters.

Let’s recap before we begin the last dash:

{% include indent_paragraph.html content=
"Recall that an artificial neuron can be succinctly described as a function that takes in $ X $ and uses its parameters $ W $ to do some computations to spit out an activation value that we expect to be close to the actual correct value (the ground truth). This also means that we expect some level of error between the activation value and the ground truth, and the loss function gives us a measure of this error in the form of single scalar value." 
%}

{% include indent_paragraph.html content=
"We want the activation to be as close as possible to the ground truth by getting the loss to be as small as possible. In order to do that, we want to find a set of values for $ W $ such that the loss is always as low as possible." 
%}

{% include indent_paragraph.html content=
"What remains to be seen is how we pull this off." 
%}

### **Gradient Descent Algorithm**
We have a loss function that is a function of the weights and biases, and we need a way to find the set of weights and biases that minimizes the loss. This is a clearcut optimization problem.

There are many ways to solve this optimization problem, but we will go with the one that scales excellently with deep neural networks, since that is the eventual goal of this writeup. And that brings us to the gradient descent algorithm.

We will illustrate how it works using a simple scenario where we have a dataset made of one feature and one target, and we want to use the mean square error as cost function. We specify a linear activation function ($$a=f(a)$$) for the neuron. Then the equation for our neuron will be:

$$
a=f\left(z\right)=w_1\ \bullet x_1+w_0
$$

Our cost function will be:

$$
J=\frac{1}{m}\bullet\sum_{j=0}^{m}{({y}_j-a_j)}^2=\frac{1}{m}\bullet\sum_{j=0}^{m}{(y_j-\ w_{1,j}\ \bullet x_{1,j}+w_{0,j})}^2
$$

Let’s further simplify our scenario by assuming we will only run computations for only one datapoint at a time.

$$
J={(Y_j-\ w_{1,j}\ \bullet x_{1,j}+w_{0,j})}^2
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

Stochastic gradient descent means that randomization is introduced during the selection of the batch of datapoints to be used in the calculations of the gradient descent. Some people will distinguish further by defining mini-batch stochastic gradient descent as when batches of datapoints are randomly selected from the dataset while stochastic gradient descent refers to just using a single randomly selected datapoint.

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

From the above equation for $$\boldsymbol{z}$$, we can immediately compute the Jacobian $$\frac{\partial z}{\partial w}$$.

The details goes like this:

$$
\frac{\partial z}{\partial w}=\left[\begin{matrix}\frac{\partial z_1}{\partial w_1}&\frac{\partial z_1}{\partial w_2}&\cdots&\frac{\partial z_1}{\partial w_n}\\\frac{\partial z_2}{\partial w_1}&\frac{\partial z_2}{\partial w_2}&\cdots&\frac{\partial z_2}{\partial w_n}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial z_m}{\partial w_1}&\frac{\partial z_m}{\partial w_2}&\cdots&\frac{\partial z_m}{\partial w_n}\\\end{matrix}\right]
$$

We can observe that the Jacobian $$\frac{\partial z}{\partial w}$$ is an $$m$$-by-$$n$$ matrix. But at this stage, our Jacobian isn't giving us anything useful because we still need the solution for each element of the matrix. But we won’t actually solve every single one of those, because it may become impractical if we had, say, a million-by-million matrix.

We’ll solve one generalized element of the Jacobian and extend the pattern to the rest. Let’s begin.

We pick an element $$\frac{\partial z_j}{\partial w_i}$$ from the matrix, and immediately we observe that we have already encountered the generalized elements $$z_j$$ and $$w_i$$ in the following equation:

$$
z_j=w_1\ \bullet x_{1,j}+w_2\ \bullet x_{2,j}+\ldots+w_n\ \bullet x_{n,j}+w_0=\sum_{i=0}^{n}{w_i\ \bullet x_{i,j}}
$$

Therefore:

$$
\frac{\partial z_j}{\partial w_i}=\frac{\partial\left(\sum_{i=0}^{n}{w_i\ \bullet x_{i,j}}\right)}{\partial w_i}
$$
The above is a partial derivative w.r.t. $$w_i$$, so we temporarily consider $$x_{i,j}$$ to be a constant.

$$
\frac{\partial z_j}{\partial w_i}=\frac{\partial\left(\sum_{i=0}^{n}{w_i\ \bullet x_{i,j}}\right)}{\partial w_i}=x_{i,j}
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
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{1}{1+e^{-z_k}}\bullet\frac{e_k^z}{e_k^z}\right)}{\partial z_j}=\frac{\partial\left(\frac{e_k^z}{e_k^z+1}\right)}{\partial z_j}
$$

The reason for this is to make the use of the [quotient rule of differentiation](https://en.wikipedia.org/wiki/Quotient_rule) for solving the derivative easier to work with.

We have to consider two possible cases. One is where $$k$$ and $$j$$ are equal, e.g. $$\frac{\partial a_2}{\partial z_2}$$, and the other is when they are not, e.g. $$\frac{\partial a_1}{\partial z_2}$$.

For $$k\neq j$$:

$$
\frac{\partial a_k}{\partial z_j}=\frac{\partial\left(\frac{e^{z_k}}{e^{z_k}+1}\right)}{\partial z_j}=0
$$

If it’s unclear how the above worked out, then recall that $$z_k$$ is a constant because we are differentiating w.r.t. $$z_j$$.

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
\frac{\partial a_k}{\partial z_j}=\frac{e^{z_k}\bullet\left(e^{z_k}+1\right)-\left(e^{z_k}\right)^2}{\left(e^{z_k}+1\right)^2}=\frac{\left(e^{z_k}\right)^2+e^{z_k}-\left(e^{z_k}\right)^2}{\left(e^{z_k}+1\right)^2}= \color{magenta}{\frac{e^{z_k}}{e^{z_k}+1}} \bullet\left(\frac{1}{e^{z_k}+1}\right)
$$

Now we clearly see the original activation function in there (in <font color="magenta">magenta</font>). But the other term also looks very similar, so we rework it a little more:

$$
\frac{\partial a_k}{\partial z_j}=\frac{e^{z_k}}{e^{z_k}+1}\bullet\left(\frac{1}{e^{z_k}+1}\right)=\color{magenta}{\frac{e^{z_k}}{e^{z_k}+1}}\bullet\left(1-\color{magenta}{\frac{e^{z_k}}{e^{z_k}+1}}\right)
$$

We can now simply substitute it in the activation (while recalling that $$k\ =\ j$$):

$$\frac{\partial a_k}{\partial z_j}=a_k\bullet\left(1-a_k\right)=a_j\bullet\left(1-a_j\right)$$

Therefore, our Jacobian becomes:

$$
\frac{\partial a}{\partial z}=\left[\begin{matrix}a_1\bullet\left(1-a_1\right)&0&\cdots&0\\0&a_2\bullet\left(1-a_2\right)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&a_m\bullet\left(1-a_m\right)\\\end{matrix}\right]
$$

It’s an $$m$$-by-$$m$$ diagonal matrix.

Two Jacobians are down and one more to go.

However, I will leave the details for the last Jacobian $$\frac{\partial J}{\partial a}$$ as an exercise for you (it’s not more challenging than the other two). Here's the setup for it.

The cost gradient $$\frac{\partial J}{\partial a}$$ depends on the choice of the cost function since it is obviously the gradient of the cost w.r.t. activation. Since we are using a logistic activation function, we will go ahead and use the logistic loss function (a.k.a. cross entropy loss or negative log-likelihoods):

$$J=-\frac{1}{m}\bullet\sum_{j}^{m}{y_j\bullet l o g{(a}_j)+(1-y_j)\bullet\log({1-a}_j)}$$

The result for $$\frac{\partial J}{\partial \boldsymbol{a}}$$ is:

$$
\frac{\partial J}{\partial\boldsymbol{a}}=-\frac{1}{m}\bullet\left(\frac{ \boldsymbol{y}}{\boldsymbol{a}}-\frac{1-\boldsymbol{y}}{1-\boldsymbol{a}}\right)
$$

Note that all the arithmetic operations in the above are all elementwise. The resulting cost gradient is a vector that has same shape as $$a$$ and $$y$$, which is $$1$$-by-$$m$$.

Now we recombine everything. Therefore, the equation for computing the cost gradient for an artificial neuron that uses a logistic activation function and a cross entropy loss is:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{a}}\frac{\partial \boldsymbol{a}}{\partial z}\frac{\partial z}{\partial \boldsymbol{w}}=-\frac{1}{m}\bullet\left(\frac{\boldsymbol{y}}{\boldsymbol{a}}-\frac{1-\boldsymbol{y}}{1-\boldsymbol{a}}\right)\frac{\partial a}{\partial z}X^T
$$

I didn’t sub in the diagonal matrix of $$\frac{\partial a}{\partial z}$$ because we can do one more thing to further compact and simplify the above equation. In fact, from here out, there are multiple ways we can choose to arrange and present these terms, and the results will all still be the same.

We choose to combine the first two gradients into $$\frac{\partial J}{\partial \boldsymbol{z}}$$ such that $$\frac{\partial J}{\partial w}$$ is:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T
$$

Therefore:

$$
\frac{\partial J}{\partial\boldsymbol{z}}=\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}
$$

Now we work on $$\frac{\partial J}{\partial \boldsymbol{z}}$$:

$$
\frac{\partial J}{\partial \boldsymbol{z}}=\color{brown}{\frac{\partial J}{\partial \boldsymbol{a}}}\color{blue}{\frac{\partial \boldsymbol{a}}{\partial z}}=\color{brown}{-\frac{1}{m}\bullet\left(\frac{ \boldsymbol{y}}{ \boldsymbol{a}}-\frac{1- \boldsymbol{y}}{1- \boldsymbol{a}}\right) }\color{blue}{\left[\begin{matrix}a_1\bullet\left(1-a_1\right)&0&\cdots&0\\0&a_2\bullet\left(1-a_2\right)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&a_m\bullet\left(1-a_m\right)\\\end{matrix}\right]}
$$

We see the multiplication of a vector with a diagonal matrix, and this is a cue for us to use elementwise multiplication (a.k.a. Hadamard product) here, which is denoted as $$\odot$$.

The mathematical relationship between Hadamard product and matrix multiplication plays out like this.

Say we have a row vector $$v$$ and a diagonal matrix $$D$$, and when we flatten the $$D$$ into a row vector $$d$$ (that is, we pull out the diagonal from $$D$$ and put it into a row vector), whose elements is just the diagonal of $$D$$, then we can write:

$$
\color{brown}{v}\color{blue}{D}=\color{brown}{v} \odot \color{blue}{d}
$$

(Test out the above for yourself with small vectors and matrices and see if the two sides indeed equate to one another).

We apply this relationship to our gradients and get:

$$
\frac{\partial J}{\partial \boldsymbol{z}}=\frac{\partial J}{\partial \boldsymbol{a}}\odot\frac{\partial \boldsymbol{a}}{\partial z}=-\frac{1}{m}\bullet\left(\frac{ \boldsymbol{y}}{ \boldsymbol{a}}-\frac{1- \boldsymbol{y}}{1- \boldsymbol{a}}\right)\ \odot(a\bullet\left(1-a\right))
$$

Where $$\frac{\partial \boldsymbol{a}}{\partial z}$$ is no longer a diagonal matrix but now a vector whose elements is from the diagonal. And $$\frac{\partial J}{\partial \boldsymbol{z}}$$ is an $$1$$-by-$$m$$ matrix.

Therefore, our final equation for computing the cost gradient $$\frac{\partial J}{\partial w}$$ is:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T=\left(-\frac{1}{m}\bullet\left(\frac{ \boldsymbol{y}}{ \boldsymbol{a}}-\frac{1- \boldsymbol{y}}{1- \boldsymbol{a}}\right)\ \odot(a\bullet\left(1-a\right))\right)X^T
$$

Now for $$\frac{\partial J}{\partial b}$$, we can borrow a lot of what we did for $$\frac{\partial J}{\partial w}$$ here as well.

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial \boldsymbol{z}}\ \frac{\partial z}{\partial \boldsymbol{b}}=\frac{\partial J}{\partial \boldsymbol{a}}\frac{\partial \boldsymbol{a}}{\partial z}\frac{\partial z}{\partial \boldsymbol{b}}
$$


Although the further breakdown of $$\frac{\partial J}{\partial\boldsymbol{z}}$$ into $$\frac{\partial J}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial z}$$ is shown above, we won’t need to use that since we already fully delineated $$\frac{\partial J}{\partial\boldsymbol{z}}$$ earlier. So, we just tackle $$\frac{\partial J}{\partial\boldsymbol{z}}\frac{\partial z}{\partial\boldsymbol{b}}$$. 

Actually just $$\frac{\partial z}{\partial\boldsymbol{b}}$$:

$$
\frac{\partial\boldsymbol{z}}{\partial\boldsymbol{b}}=\frac{\partial(\boldsymbol{wX}\ +\ \boldsymbol{b})}{\partial\boldsymbol{b}}=\frac{\partial(\boldsymbol{wX})}{\partial\boldsymbol{b}}+\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}=0+\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}=1
$$

Therefore:

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial\boldsymbol{z}}
$$

We just cut corners in the above by doing $$\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}=1$$. How? Well, the equivalence of $$\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}$$ to 1 only holds by assuming that $$b$$ is always a scalar, which organically it is (it is simply the weight for the bias node). But we can’t forget that during computation, $$b$$ gets broadcasted to match the shape of $$z$$.

Logically, $$\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}$$ must always have same shape as $$b$$. This means that during computation, just like $$b$$, $$\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}$$ should get broadcasted into a tensor of ones with same shape as $$z$$.

That is, we should have this:

$$
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial\boldsymbol{z}}\ \frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}
$$

{% include indent_paragraph.html content=
"Where $ \frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}} $ is a tensor of ones (a vector in the case of just one artificial neuron) with same shape as $ z $; so, think of something like this $ \left[\begin{matrix}1&1&\cdots&1\\\end{matrix}\right] $."
%}

Remember that even though $$b$$ is a scalar (or $$1$$-by-$$1$$ vector), it gets broadcasted into a 1-by-$$m$$ vector during the forward pass. We must keep in mind that $$b$$ is a parameter of the estimator, and it would be very counterproductive to define it in a way that binds it to the number of examples (datapoints) in a batch. This is why its fundamental form is a scalar.

As mentioned earlier, matrix multiplication, or specifically vector-matrix multiplication, is essentially one example of tensor contraction. 

Here is a quick overview of tensor contraction.

From the perspective of tensor contraction, the vector-matrix multiplication of a row vector $$v$$ and a matrix $$M$$ to produce a row vector $$u$$ is:

$$u_q=\sum_{p}{v_p\bullet M_{p,q}}$$

Where the subscript $$p$$ tracks the only non-unit axis of the vector $$v$$, and the subscript $$q$$ tracks second axis of the matrix $$M$$.


Here is an example to illustrate the above. Say that $$v$$ and $$M$$ are:

$$
v=\ \left[\begin{matrix}1&2\\\end{matrix}\right]
$$

$$
M=\left[\begin{matrix}3&5&7\\4&6&8\\\end{matrix}\right]
$$

{% include indent_paragraph.html content=
"The vector $ v $ is $ 1 $-by-$ 2 $, and we will use the subscript $ q $ to track the non-unit axis, i.e. the second axis (the one that counts to a maximum of 2). That is: $ v_1=1 $ and $ v_2=2 $"
%}

{% include indent_paragraph.html content=
"The matrix $ M $ is $ 2 $-by-$ 3 $, and we will use the subscript $ q $ to track the first axis (the one that counts to a maximum of 2) and $ p $ to track the second axis (the one that counts to a maximum of 3). That is $ M_{2,1}=4 $ and $ M_{1,3}=7 $."
%}

We know that the vector-matrix multiplication, $$vM$$, produces a vector. Let’s call it $$u$$, and it has the shape $$1$$-by-$$3$$.

$$
u=\left[\begin{matrix}u_1&u_2&u_3\\\end{matrix}\right]
$$

Using the tensor contraction format, we can fully characterize what the resulting vector $$u$$ is, by describing it elementwise:

$$
u_q=\sum_{p}{v_p\bullet M_{p,q}}
$$

For instance,

$$
u_1=v_1\bullet M_{1,1}+v_2\bullet M_{2,1}=1\bullet3+2\bullet4=11
$$

And we can do this for $$u_2$$ and $$u_3$$ (try it). In all, we have:

$$
u=\left[\begin{matrix}11&17&23\\\end{matrix}\right]
$$

To summarize, the vector multiplication $$vM$$ is a contraction along the axis tracked by subscript $$p$$.

We can use the tensor contraction format to more properly delineate $$\frac{\partial J}{\partial\boldsymbol{b}}$$ without cutting corners.

In tensor contraction format, $ \frac{\partial J}{\partial b} $ is:

$$
\frac{\partial J}{\partial b}=\sum_{j=1}^{m}{\left(\frac{\partial J}{\partial\boldsymbol{z}}\right)_j\bullet\left(\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}\right)_j}
$$

And because $$\frac{\partial\boldsymbol{b}}{\partial\boldsymbol{b}}$$ is a vector of ones, we have:

$$
\frac{\partial J}{\partial b}=\sum_{j=1}^{m}\left(\frac{\partial J}{\partial\boldsymbol{z}}\right)_j
$$

In essence, we summed across the second axis of $ \frac{\partial J}{\partial z} $ which reduced it to a $1$-by-$1$ vector, which we then equated to $ \frac{\partial J}{\partial b} $.

We now have all our cost gradients fully delineated.

<table>
<td>
<details>
<summary>
<b>The intuition of the cost gradient with respect to bias </b>
</summary>
<p>
The intuition of this summing is that we are averaging across all the datapoints. We are taking the average of the gradients for all the datapoints in the batch and set that as our gradient $ \frac{\partial J}{\partial b} $.
<br><br>
Note that the mathematical operation of averaging is simply the summation of terms divided by the number of terms being summed.
<br><br>
We are contracting along the first axis, represented by subscript $ j $, which is also the axis that tracks the datapoints. In other words, we are summing across the datapoints. This is also the first step in an averaging operation. 
<br><br>
The next and final step in the averaging is the division of the summation by the number of terms that we are summing up, which would be the batch size (the number of datapoints in the batch). So, what about that?
<br><br>
Well that was already baked in when we looked into $ \frac{\partial J}{\partial \boldsymbol{z}}=\frac{\partial J}{\partial \boldsymbol{a}}\frac{\partial a}{\partial \boldsymbol{z}} $. The term $ \frac{1}{m} $ in $ \frac{\partial J}{\partial \boldsymbol{a}} $ is it.
<br><br>
So, to summarize, the operations involved in the summing of $ \frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T $ along its second axis to produce $ \frac{\partial J}{\partial \boldsymbol{b}} $ is analogous to it taking the average, across all datapoints in the batch, of the cost gradient associated with each datapoint. Also, the same kind of intuition can be drawn for $ \frac{\partial J}{\partial w} $.
</p>
</details>
</td>
</table>

Although, we don't really need to see the equation for $ \frac{\partial J}{\partial w} $ in its contraction format, we will present it for thee sake of it. We already know that $ \frac{\partial J}{\partial w} $ is:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T
$$

And we also already know that $ \frac{\partial J}{\partial \boldsymbol{z}} $ is a $ 1 $-by-$ m $ row vector and $ X $ is an $ n $-by-$ m $ matrix, which makes $ X^T $ an $ m $-by-$ n $ matrix. In tensor contraction format, the above equation is:

$$
\left(\frac{\partial J}{\partial \boldsymbol{w}}\right)_i=\sum_{j}{\left(\frac{\partial J}{\partial \boldsymbol{z}}\right)_j\bullet\left(X^T\right)_{j,i}}
$$

### **Summary of workflow**

The workflow is as follows:

#### **Initialize parameters**

We first initialize our parameters, and we will do this randomly.

However, there are a handful of initialization schemes out there that promise to initialize our parameters with values that enable the network to be optimized faster. You can think of it as beginning a journey from a point closer to the destination, which allows you to finish faster than another person who began much farther away because they chose to randomly begin at some arbitrary point.

One such scheme for sigmoid activation functions (e.g. logistic, hyperbolic tangent, etc.), the Xavier Initialization Scheme (introduced by Xavier Glorot in 2010), is implemented in my code for deep neural networks.

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
J=-\frac{1}{m}\bullet\sum_{j}^{m}{y_i\bullet l o g{(y}_i)+(1-a_i)\bullet\log({1-a}_i)}
$$

But we will not directly use this cost any further in this workflow. We just need it in order to keep track of how good how network is doing.

#### **Optimize parameters**

Now we will optimize our parameters in such a way that our loss decreases. The backward pass begins here. We start by first computing the cost gradient $$\frac{\partial J}{\partial w}$$:

$$
\frac{\partial J}{\partial w}=\ \frac{\partial J}{\partial \boldsymbol{z}}X^T=\left(-\frac{1}{m}\bullet\left(\frac{ \boldsymbol{y}}{ \boldsymbol{a}}-\frac{1- \boldsymbol{y}}{1- \boldsymbol{a}}\right)\ \odot(a\bullet\left(1-a\right))\right)X^T
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