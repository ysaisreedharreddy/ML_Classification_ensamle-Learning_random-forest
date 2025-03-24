Random Forest Classification
This repository contains a Python implementation of a Random Forest Classifier designed to predict user behavior based on social network advertisement data. The model uses age and estimated salary as features to predict whether users have purchased a product.

Features
Data Preprocessing: Script includes preprocessing of the dataset such as feature extraction and data splitting.
Random Forest Model: Utilization of the RandomForestClassifier from scikit-learn with configured hyperparameters.
Model Evaluation: Includes a confusion matrix and accuracy score to evaluate the performance.
Visualization: Function to visualize the decision boundaries created by the Random Forest model on both the training and test sets.

Dataset
The dataset (Social_Network_Ads.csv) consists of user information like Age and Estimated Salary alongside a Purchase flag which indicates whether the product was bought or not. This dataset is split into training and testing sets to train the model and then evaluate its performance.

Usage
To run this script, ensure you have Python installed along with the necessary libraries:
NumPy
pandas
matplotlib
scikit-learn

This will execute the script, train the model, and output the performance metrics and visualizations.

Results
Outputs include:
Confusion Matrix: To visualize the accuracy of predictions.
Accuracy Score: Overall accuracy of the model on the test data.
Bias and Variance: Indicators of the model's performance across training and test datasets.
