# Import necessary libraries for data handling and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset from a specified CSV file into a pandas DataFrame
dataset = pd.read_csv(r"C:\Users\prasu\DS2\git\classification\6. Ensamble_learning\5. RANDOM FOREST\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values  # Extracting features such as Age and Estimated Salary
y = dataset.iloc[:, -1].values      # Extracting the target variable (Purchased)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the Random Forest Classification model on the Training set
classifier = RandomForestClassifier(max_depth=4, n_estimators=60, random_state=0, criterion='entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Generating and printing the confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
ac = accuracy_score(y_test, y_pred)
print('Accuracy:', ac)

# Evaluating the model's training and test performance
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)
print('Bias (Training Score):', bias)
print('Variance (Test Score):', variance)

# Visualization function for training and test sets
def visualize_results(X_set, y_set, set_description, classifier):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1)
    )
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=f"Class {j}")
    plt.title(f'Random Forest Classification ({set_description} set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

# Example usage:
visualize_results(X_train, y_train, 'Training', classifier)
visualize_results(X_test, y_test, 'Test', classifier)
