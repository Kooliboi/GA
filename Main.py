import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Importera MNIST dataset
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# Omvandla dataset till en numpy array
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Dela upp datan i träning och prov. Vissa kommer att användas för att träna på, och andra för att prova nätverket
data_test = data[:1000].T
Y_test = data_test[0].reshape(-1)
X_test = data_test[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0].reshape(-1)
X_train = data_train[1:n] / 255.0

# One-hot encoding (binärt omvandla)
num_classes = 10
Y_train_one_hot = np.eye(num_classes)[Y_train]
Y_test_one_hot = np.eye(num_classes)[Y_test]
    


# Neurala Nätverket
class Layers:
    def __init__(self, n_inputs, n_neurons):
        self.w = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.b = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.w) + self.b
        return self.output
    def backward(self, dvalues):
        self.dw = np.dot(self.inputs.T, dvalues)
        self.db = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.w.T)
        return self.dinputs

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    def backward(self, dvalues):
        dvalues = dvalues.copy()
        dvalues[self.inputs <= 0] = 0
        return dvalues

class ActivationSoftmaxLoss:
    def forward(self, inputs, y_true):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        y_pred = np.clip(self.output, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=1)
        else:
            correct_confidences = y_pred[np.arange(len(y_pred)), y_true]
        self.loss = -np.log(correct_confidences).mean()
        return self.loss
    def backward(self, y_true):
        samples = len(self.output)
        self.dinputs = self.output - y_true
        self.dinputs /= samples
        return self.dinputs

class GradientDescent:
    def __init__(self, learning_rate=0.005):
        self.learning_rate = learning_rate
    def update(self, layer):
        layer.w -= self.learning_rate * layer.dw
        layer.b -= self.learning_rate * layer.db


# Aktivera alla funtioner för att fungera som ett neuralt nätverk
input_size = 784
hidden1_size = 128
hidden2_size = 64
output_size = 10

layer1 = Layers(input_size, hidden1_size)
activation1 = ActivationReLU()
layer2 = Layers(hidden1_size, hidden2_size)
activation2 = ActivationReLU()
layer3 = Layers(hidden2_size, output_size)
activation_loss = ActivationSoftmaxLoss()
optimizer = GradientDescent(learning_rate=0.005)

# Loop
epochs = 1000
batch_size = 128
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    for i in range(0, X_train.shape[1], batch_size):
        X_batch = X_train.T[i:i + batch_size]
        Y_batch = Y_train_one_hot[i:i + batch_size]
        
        l1_output = layer1.forward(X_batch)
        l1_activated = activation1.forward(l1_output)
        l2_output = layer2.forward(l1_activated)
        l2_activated = activation2.forward(l2_output)
        l3_output = layer3.forward(l2_activated)
        loss = activation_loss.forward(l3_output, Y_batch)

        
        dloss = activation_loss.backward(Y_batch)
        dL3 = layer3.backward(dloss)
        dL2_activated = activation2.backward(dL3)
        dL2 = layer2.backward(dL2_activated)
        dL1_activated = activation1.backward(dL2)
        dL1 = layer1.backward(dL1_activated)
        
        optimizer.update(layer1)
        optimizer.update(layer2)
        optimizer.update(layer3)



for epoch in range (epochs):
    layer1.forward(X_test.T)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    layer3.forward(activation2.output)
    predictions = np.argmax(layer3.output, axis=1)
    accuracy = np.mean(predictions == Y_test)
            
    loss_history.append(loss)
    accuracy_history.append(accuracy)
    
print(f"Testing Results - Epoch {epoch+1}/{epochs} - Loss: {loss:.10f} - Accuracy: {accuracy:.10f}")

    
