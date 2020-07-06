# Machine-Learning-Intro
Covers Machine Learning basics.

<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/Machine-Learning-image.png" height="300" width="100%" ></a>
**Machine learning** is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

## Three types of machine learning: 
<a href="url"><img src="https://github.com/RocqJones/Machine-Learning-Intro/blob/master/imgs/Types-of-Machine-Learning-algorithms.jpg" height="400" width="100%" ></a>
### 1. Supervised learning.
### 2. Unsupervised learning.
### 3. Reinforcement learning.

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
