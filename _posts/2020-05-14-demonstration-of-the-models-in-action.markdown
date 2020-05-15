---
layout: post
title:  "Catching AI with its pants down: Demonstration of the Models in Action"
logline: "Demonstrating the models that were designed from scratch by tackling real, research datasets."
date:   "2020-05-14"
categories: machine-learning
permalink:
comments: true
---
{% include scripts.html %}

{% include blogseries_mantra_catching_ai.html %}

* TOC
{:toc}


## **Prologue**

This is part 8 of this blog series, *Catching AI with its pants down*. This blog series aims to explore the inner workings of neural networks and show how to build a standard feedforward neural network from scratch.

This part is simply a central page for all the Jupyter Notebooks that showcase the artificial neuron model and the neural network model in action. Both models were designed straight from the mathematical formulations, and the complete information on how are detailed in this blog series.

Except for one demonstration with a toy dataset created for pedagogical purposes for this blog series, all the other demonstrations are done with real-world, research datasets.

{% include blogseries_index_catching_ai.html %}

## **Demonstrations for the Neural Network Model**

These are the demonstrations carried out for the model that involves just a network of artificial neuron.

### **US Adult Income Dataset**

The US adult income dataset contains data on 48842 individuals, and includes fields containing data on their age, education, sex, income level (classed into 2 categories: over 50k or under 50k), etc.

The model (a neural network) learned to reach a **78% accuracy** in predicting whether an individual makes over 50k a year or not, based on just data from other fields.

Full details can be found in [**this Jupyter notebook**](https://github.com/princyok/deep_learning_without_ml_libraries/tree/master/neural_network/US_Adult_Income.ipynb){:target="_blank"}.

### **Chest X-Ray Pneumonia Image Dataset**

This is a dataset containing 4273 chest x-ray images of children suffering from pneumonia (both bacterial and viral) and 1592 chest x-ray images of those not suffering from the lung infection. 

The model (a neural network) learned to reach a **92% accuracy** in predicting whether a patient has pneumonia or not, based on their chest x-ray image.

Full details can be found in [**this Jupyter notebook**](https://github.com/princyok/deep_learning_without_ml_libraries/tree/master/neural_network/Chest_XRay_Pneumonia.ipynb){:target="_blank"}.

### **Cat vs Not-Cat Dataset**

This is a dataset of images of cats and those without cats. 

The model (a neural network) learned to reach a **73% accuracy** in predicting whether there is a cat in an image or not.

***Will be released soon.***


## **Demonstrations for the Single Artificial Neuron Model**

These are the demonstrations done for the model that involves just a single artificial neuron.

### **Breast Cancer Wisconsin Dataset**

The Breast cancer Wisconsin dataset is a dataset that describe the characteristics of the cell nuclei for 569 breast lumps, and includes whether they are malignant or benign. Published by Clinical Science Center, University of Wisconsin.

The model (a single artificial neuron) was able to reach a **92% accuracy** in distinguishing whether the tumour is malignant or benign.

Full details can be found in [**this Jupyter notebook**](https://github.com/princyok/deep_learning_without_ml_libraries/tree/master/one_neuron/Breast_Cancer_Wisconsin.ipynb){:target="_blank"}.

### **Toy dataset**
The toy dataset was originally introduce in [part 2](/understand-an-artificial-neuron-from-scratch.html#toy-dataset-for-this-blog-series){:target="_blank"} of this blog series. It's a dataset simulated using [classical mechanics](https://en.wikipedia.org/wiki/Kinetic_energy) and random uniform noise. It had two features, `velocity` and `mass`, and a categorical binary output, `energy level`.

The model (a single artificial neuron) was able to reach a **89% accuracy** in distinguishing whether the motion of a ball, characterized by its velocity and mass, was high or low energy level.

Full details can be found in [**this Jupyter notebook**](https://github.com/princyok/deep_learning_without_ml_libraries/tree/master/one_neuron/Toy_Dataset_vs_Artificial_Neuron.ipynb){:target="_blank"}.