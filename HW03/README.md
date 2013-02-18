Classifiers
==============

Implementation of the following classifiers:

*   Naïve Bayes
*   Logistic Regression
*   Perceptron

Requirements:
-------------

Python 3.X

Usage:
------

To run the Naïve Bayes Classifier, execute the following from the root project directory:
        
    python -m classify.main naive_bayes <train> <test> <beta> <model>
   
To run the Logistic Regression Classifier, execute the following from the root project directory:
        
    python -m classifiers.main logistic <train> <test> <eta> <sigma> <model>
    
To run the Perceptron Classifier, execute the following from the root project directory:
        
    python -m classify.main perceptron <train> <test> <eta> <model>
