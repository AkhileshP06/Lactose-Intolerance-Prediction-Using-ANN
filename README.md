
### Lactose Intolerance Prediction Using Artificial Neural Networks

This repository contains compilation of my solutions for various LeetCode questions. I use LeetCode as a platform to practice Data Structures and Algorithms  and I document my solutions here for reference.


## Project Overview
This project was developed as a requirement for the course __21 BIO 201 Intelligence of Biological Systems 2__  at Amrita School of Engineering, Bangalore. The goal of this project is to predict lactose intolerance using an artificial neural network (ANN) by analyzing the LCT gene's nucleotide sequences.
## Abstract

The project presents an artificial neural network model that predicts lactose intolerance. The input consists of nucleotide sequences of the LCT gene, which are key indicators of lactose tolerance/intolerance. The ANN model is designed to assist specialists in predicting lactose intolerance, reducing the time required for diagnosis.
## Features

 - LCT Gene Analysis: The model focuses on the genotype of LCT 13910 C/T and LCT 22018 G/A polymorphisms, which are reliable predictors of lactose tolerance/intolerance.
 - Artificial Neural Network: Utilizes a neural network with 8 input parameters, each representing a single nucleotide, to make predictions.
 - Python Implementation: The entire model is implemented in Python, making use of libraries like NumPy.

## Program Flow
1. __Initialization__
The program begins by defining a class called `NeuralNetwork`. When an instance of this class is created, it initializes the following:

 - Weights: Randomly generated weights, which are parameters that the network will adjust during training to minimize prediction errors.
 - Bias: A randomly generated bias term that is added to the weighted sum of inputs before applying the activation function.
 - Learning Rate: A predefined constant that controls how much the weights and bias should be updated during training.

2. __Sigmoid Function__
The neural network uses a sigmoid function as the activation function. This function squashes input values to a range between 0 and 1, which is useful for binary classification tasks like predicting lactose intolerance.

 - `_sigmoid(x)`: This private method computes the sigmoid of the input `x`.

```python
def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
```
 - `_sigmoid_deriv(x)`: This private method computes the derivative of the sigmoid function. The derivative is used during backpropagation to update the weights and bias.

```python
def _sigmoid_deriv(self, x):
    return self._sigmoid(x) * (1 - self._sigmoid(x))
```

3. __Prediction__
The `predict` method takes an input vector (representing the LCT gene sequence) and passes it through the neural network to produce an output, which is a prediction of whether the person is lactose tolerant or intolerant.

 - `predict(input_vector)`: 
Calculates the dot product of the input vector and the weights, adds the bias, and applies the sigmoid function to produce a prediction.

```python
def predict(self, input_vector):
    layer_1 = np.dot(input_vector, self.weights) + self.bias
    prediction = self._sigmoid(layer_1)
    return prediction
```

4. __Gradient Computation__
To train the neural network, the program must adjust the weights and bias to minimize the error between the predicted output and the actual target output. This is done using gradient descent.

 - `_compute_gradients(input_vector, target)`:
Computes the error between the prediction and the target output and the gradient of the error with respect to the weights and bias.

```python
def _compute_gradients(self, input_vector, target):
    layer_1 = np.dot(input_vector, self.weights) + self.bias
    prediction = self._sigmoid(layer_1)

    derror_dprediction = 2 * (prediction - target)
    dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
    dlayer1_dbias = 1
    dlayer1_dweights = input_vector

    derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
    derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

    return derror_dbias, derror_dweights
```
5. __Parameter Update__
After computing the gradients, the program updates the weights and bias to reduce the prediction error.

 - `_update_parameters(derror_dbias, derror_dweights)`:
Updates the weights and bias by subtracting the product of the gradient and the learning rate from the current weights and bias.

```python
def _update_parameters(self, derror_dbias, derror_dweights):
    self.bias -= derror_dbias * self.learning_rate
    self.weights -= derror_dweights * self.learning_rate
```

6. __Training__
The train method is responsible for repeatedly updating the weights and bias over many iterations to minimize the prediction error.

 - `train(input_vectors, targets, iterations)`:
The method iterates over the training data, computes the gradients, and updates the weights and bias. Every 100 iterations, the cumulative error is measured and stored.
```python
def train(self, input_vectors, targets, iterations):
    cumulative_errors = []
    for current_iteration in range(iterations):
        # Pick a data instance at random
        random_data_index = np.random.randint(len(input_vectors))

        input_vector = input_vectors[random_data_index]
        target = targets[random_data_index]

        # Compute the gradients and update the weights
        derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)
        self._update_parameters(derror_dbias, derror_dweights)

        # Measure the cumulative error for all the instances
        if current_iteration % 100 == 0:
            cumulative_error = 0
            for data_instance_index in range(len(input_vectors)):
                data_point = input_vectors[data_instance_index]
                target = targets[data_instance_index]
                prediction = self.predict(data_point)
                error = np.square(prediction - target)
                cumulative_error += error
            cumulative_errors.append(cumulative_error)
            
    return cumulative_errors
```

7. __Final Prediction__
After training the neural network, a new LCT gene sequence is provided as input, and the `predict` method is used to determine whether the individual is lactose tolerant or intolerant.

The input sequence is converted into a numeric format, and the trained neural network predicts the outcome.
## Authors
- [@AkhileshP06](https://github.com/AkhileshP06)
- [@Amalkrishnaaa](https://github.com/Amalkrishnaaa)
- [@Karthick-7014](https://github.com/Karthick-7014)

