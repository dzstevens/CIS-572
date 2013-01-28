CIS-572: HW #1
==============

An implementation of an ID3 decision tree.

Requirements:
-------------

Python 3.X with Numpy installed

Usage:
------
   
To run, execute the following command from the root project directory:
        
    python -m id3.main.py <training_input_file> <test_input_file> <model_output_file>
    
You will be presented with a menu to choose the p-value for the chi-squared test. 
After you make your selection, the classifier will train on the training data 
provided by <training_input_file>, output the decision tree model to 
<model_output_file> and then attempt to label the test data in <test_input_file>. 
When finished, it will output the accuracy it achieved on the test data.

