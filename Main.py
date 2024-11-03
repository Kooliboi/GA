import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

#Numpy är en inbyggd databas som tillåter användningen av advancerad matte som matriser och vektorer
#Nnfs är en data bas  med lite info om neurala nätverk som INPUT DATA vilket är spiral data i detta fall








# inputs = X because it makes sense
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(100, 3)



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs): 
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)

activation1 = Activation_ReLU()


layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)










'''

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, 0.23], 
           [0.25, 0.92, -0.43, 0.91]]

biases = [2, 3, 0.5]

dotoutput = np.dot(inputs, np.array(weights).T) + biases

#output1 = dotoutput[0] + biases[0]
#output0 = weights[0][0]*input[0]+weights[0][1]*input[1]+weights[0][2]*input[2]+weights[0][3]*input[3] 

print(dotoutput)

'''

 
