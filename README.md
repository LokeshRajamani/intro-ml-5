# Ex-5 :Implementation-of-Logistic-Regression-Using-Gradient-Descent
# Date: 01.04.202
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

2.Load Dataset: Load the dataset using pd.read_csv.

3.Remove irrelevant columns (sl_no, salary).

4.Convert categorical variables to numerical using cat.codes.

5.Separate features (X) and target variable (Y).

6.Define Sigmoid Function: Define the sigmoid function.

7.Define Loss Function: Define the loss function for logistic regression.

8.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

9.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

10.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

11.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

12.Predict placement status for a new student with given feature values (xnew).

13.Print Results: Print the predictions and the actual values (Y) for comparison.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: LOKESH R
RegisterNumber: 212222240055 
*/
```

```
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

#display dataset
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#display dependent variables
Y

theta = np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def loss(theta,X,y):
  h = sigmoid(X.dot(theta))
  return -np.sum(y * np.log(h) + (1-y) * np.log(1 - h))


def gradient_descent(theta,X,y,alpha,num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y)/m
    theta -= alpha * gradient
  return theta

theta = gradient_descent(theta, X,y,alpha = 0.01,num_iterations=1000)

def predict(theta,X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5, 1, 0)
  return y_pred

y_pred = predict(theta, X)


accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:",accuracy)


print(y_pred)

print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

```


## Output:

Dataset:

![324399608-b759be5c-9ff7-44ab-ba18-0f95fef59a0a](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/9f2e7c30-47be-4020-a62d-294526a84c6a)

Dataset.dtypes:

![324399641-0ac77d57-1afe-49a7-aa8d-4414f90e9e92](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/6feb2bc1-ed4d-45d4-9c45-00a06af9606f)

Dataset:

![324399665-730e6d8a-e533-43b2-9d7a-71428c7ab09a](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/afb0aa16-22f3-46bb-884f-1ce1a77cbc60)

Y:

![324399781-c5a2dc74-8f83-46f5-8c99-ca4fc9fba3c3](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/a2faa436-4e38-4be7-beec-56dd1d92c633)

Accuracy:

![324679436-6ab36fdc-19a4-4d7f-ab06-cce7d3dae7df](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/2f7ebe16-185e-4f8d-ba17-3160a8cfadcc)


y_pred:

![324399812-6dce3b27-0063-401d-b7e4-5f8dfb501496](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/4591ef34-8a1c-4018-9a20-c2ef9efa1403)


Y:

![324399836-cbc84213-853e-4afc-b9f9-20a48dcd8903](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/74e50b47-024b-4d57-bb7b-e51333476774)


y_prednew:

![324399868-fcfeba87-3f2f-4d1d-b625-b63326f7da16](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/2caf50ee-ab26-4f92-876d-852f7f9a9954)


y_prednew:

![324401796-244c54d9-fd43-43e7-8432-e8ca14255d6c](https://github.com/LokeshRajamani/intro-ml-5/assets/120544804/975c2f31-ba8d-4adc-b0f0-b71400133f65)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

