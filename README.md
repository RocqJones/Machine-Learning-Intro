# Machine-Learning-Intro
Covers Machine Learning basics.

<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/Machine-Learning-image.png" height="300" width="100%" ></a>
**Machine learning** is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

## Machine Learning road-map.
## Three types of machine learning: 
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/Types-of-Machine-Learning-algorithms.jpg" height="400" width="100%" ></a>
### 1. Supervised learning.
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/supervised-learning.png" height="400" width="100%" ></a>

### 2. Unsupervised learning.
Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning method is cluster analysis, which is used for exploratory data analysis to find hidden patterns or grouping in data.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/unsupervised-learning.png" height="400" width="100%" ></a>

#### Types of unsupervised learning.
##### Clustering or Cluster analysis
Clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/k-means-clustering.png" height="400" width="100%" ></a>
* **K-means Clustering:** A method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
    ###### [K-means Clustering visualization tool to play around with](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
* **Feature Scaling:** is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/feature-scaling.jpg" height="400" width="100%" ></a>

### Supervised learning vs Unsupervised learning.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/supVsUnsup.png" height="400" width="100%" ></a>

### 3. Reinforcement learning.
Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/reinforcement-learning.jpg" height="400" width="100%" ></a>

## [scikit-learn library](https://scikit-learn.org/stable/)
**Scikit-learn** (formerly scikits.learn and also known as sklearn) is a free software machine learning library for the Python programming language. It features various classification; 
* regression and clustering algorithms including support vector machines,
* random forests, 
* gradient boosting, 
* k-means and DBSCAN, and is designed to *interoperate with the Python numerical and scientific libraries NumPy and SciPy.
    - To install do: ```$ pip install -U scikit-learn```
    - Confirm installation with the following:
      ```
      python -m pip show scikit-learn
      python3 -c "import sklearn; sklearn.show_versions()"
      ```
## Algorithms.
### 1. Naive Bayes Rule algorithm.
It is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

#### [Mini-project 1](https://github.com/RocqJones/Machine-Learning-Intro/3.Mini-Project-1)
*We’ll do something very similar in this project. We have a set of emails, half of which were written by one person and the other half by another person at the same company . Our objective is to classify the emails as written by one person or the other based only on the text of the email. We will start with Naive Bayes in this mini-project, and then expand in later projects to other algorithms.*
##### Requirements:
* **[Install sklearn](https://scikit-learn.org/stable/install.html)**
* **[Install natural language toolkit](https://pypi.org/project/nltk/)**

### 2. Support Vector Machine (SVM) algorithm.
A Support Vector Machine (SVM) performs classification by finding the hyperplane that maximizes the margin between the two classes. The vectors (cases) that define the hyperplane are the support vectors.

### 3. Decision Tree algorithm.
Decision Tree algorithm belongs to the family of supervised learning algorithms. The goal of using a Decision Tree is to *create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data)*.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/dcsion.png" height="400" width="100%">

### 4. K-nearest Neighbors algorithm.
K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).
* Classic, simple, and easy to understand.

### 5. AdaBoost, short for “Adaptive Boosting” algorithm.
It focuses on classification problems and aims to convert a set of weak classifiers into a strong one. 
* The final equation for classification is as follows;
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/adaboost.jpeg" height="200" width="100%">
* ```f_m``` stands for the ```m_th weak classifier``` and ```theta_m``` is the corresponding *weight*. 
* It is exactly the weighted combination of M weak classifiers.
    
### 6. Random Forest algorithm.
The random forest is a classification algorithm consisting of many decisions trees. 
* It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree. Check illustration below.
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/random-forest-algorithm.png" height="500" width="100%">

## Text feature extraction.
### The Bag of Words representation
Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length. In order to address this; 
* Scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:
    - tokenizing strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
    - counting the occurrences of tokens in each document.
    - normalizing and weighting with diminishing importance tokens that occur in the majority of samples / documents.

In this scheme, features and samples are defined as follows:
* each individual **token occurrence frequency** (normalized or not) is treated as a **feature**.
* the vector of all the token frequencies for a given **document** is considered a multivariate **sample**.

## Useful Machine Learning Datasets.
* [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/)
