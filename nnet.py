import numpy as np

# Created a class layer with forward propagation, backward propagation not implemented
class Layer:
    def __init__(self):
        pass
        
    def forwardprop(self, ipdata):
        raise NotImplementedError

    def backwardprop(self, operr, lrate):
        raise NotImplementedError

# Created a class FCLayer that inherits from the class Layer and randomly creates the weights and bias according to the input (ipsize) and output size (opsize)
class FCLayer(Layer):
    def __init__(self, ipsize, opsize):
        self.weights = np.random.randn(ipsize, opsize) * np.sqrt(1 / ipsize)
        self.biases = np.zeros((1, opsize))

    # The forwardprop function takes input data (ipdata) and return the calculation A = X . H + B
    def forwardprop(self, ipdata):
        self.input = ipdata
        return np.dot(ipdata, self.weights) + self.biases

    # The backwardprop function takes operr (ouput error) and lrate which is the learning rate
    def backwardprop(self, operr, lrate):
        # From output error we are calculating the input error (iperr) using dEdy . H.T
        iperr = np.dot(operr, self.weights.T)
        # And the weight error (wgterr) using input.T . dEdy
        wgterr = np.dot(self.input.T, operr)
        # Updating the weights and Bias using the gradient and learning rate
        self.weights = self.weights.astype('float64') - lrate * wgterr
        self.biases = self.biases.astype('float64') - lrate * np.sum(operr, axis=0, keepdims=True)
        return iperr

# The following are the activation functions and their derivatives
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

# Derivative for sigmoid
def sigder(x):
    zprime = sigmoid(x) * (1 - sigmoid(x))
    return zprime

def relu(x):
    z = np.maximum(0, x)
    return z

# Derivative for relu
def reluDer(x):
    zprime = np.where(x > 0, 1, 0)
    return zprime

# Leaky Relu
def lrelu(x, alpha=0.01):
    z = np.where(x > 0, x, alpha * x)
    return z

# Derivative for leaky relu
def lreluDer(x, alpha=0.01):
    zprime = np.where(x > 0, 1, alpha)
    return zprime

def tanh(x):
    z = np.tanh(x)
    return z

# Derivative of tanh
def tanhder(x):
    zprime = 1 - np.tanh(x) ** 2
    return zprime

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    z = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return z

# Derivative of Softmax
def softmaxder(x):
    s = softmax(x)
    zprime = s * (1 - s)
    return zprime

class ActivationLayer(Layer):
    # The following dictonary stores all the mapping of the name and functions
    ACTIVATIONS = {
        "sigmoid": (sigmoid, sigder),
        "relu": (relu, reluDer),
        "lrelu": (lrelu, lreluDer),
        "tanh": (tanh, tanhder),
        "softmax": (softmax, softmaxder),
    }

    def __init__(self, actname):
        # If the name of the activation (actname) is not in the activation functions, it will return an error.
        if actname not in self.ACTIVATIONS:
            raise ValueError(f"Invalid activation function '{actname}'.")
        self.activation, self.actder = self.ACTIVATIONS[actname]

    def forwardprop(self, ipdata):
        #This function just uses activation function and gives output
        self.input = ipdata
        return self.activation(ipdata)

    def backwardprop(self, operr, lrate):
        # This function takes in the output error and multiplies with the derivative of the activation function on input data.
        return self.actder(self.input) * operr

def mse(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    z = np.mean((ytrue - ypred) ** 2)
    return z

# MSE Derivative
def mseder(ytrue, ypred):
    zprime = 2 * (ypred - ytrue) / ytrue.shape[0]
    return zprime

# Binary Cross Entropy
def bcentropy(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    z = -np.mean(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))
    return z

# Binary Cross Entropy Derivative
def bcentropyDer(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    zprime = (ypred - ytrue) / (ypred * (1 - ypred) + 1e-9)
    return zprime

# Categorical Cross Entropy
def ccentropy(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    z = -np.sum(ytrue * np.log(ypred)) / ytrue.shape[0]
    return 

# Categorical Cross Entropy Derivative
def ccentropyDer(ytrue, ypred):
    zprime = ypred - ytrue
    return zprime

class Network:
    def __init__(self):
        # Initializing the layers, loss function and loss function derivative
        self.layers = []
        self.lossfunc = None
        self.lossder = None

    def add(self, layer):
        # Adding layers to the network
        self.layers.append(layer)

    def selectloss(self, ytrue, ypred):
        #Selecting the appropriate loss function through the shape of the output data
        if ytrue.shape[1] == 1:
            self.lossfunc = bcentropy
            self.lossder = bcentropyDer
        elif np.allclose(np.sum(ypred, axis=1), 1):
            self.lossfunc = ccentropy
            self.lossder = ccentropyDer
        else:
            self.lossfunc = mse
            self.lossder = mseder

    def predict(self, ipdata):
        #Predicting the output through forward propagation
        result = ipdata
        for layer in self.layers:
            result = layer.forwardprop(result)
        return result

    def fit(self, Xtrain, Ytrain, Xval, Yval, epochs, lrate, loss=None):
        # Fitting the training data and testing on validation data
        if loss:
            # if loss is given then it selects the function from the dictionary
            lossfuns = {
                "mse": (mse, mseder),
                "bcentropy": (bcentropy, bcentropyDer),
                "ccentropy": (ccentropy, ccentropyDer)
            }
            if loss not in lossfuns:
                raise ValueError(f"Unknown loss function '{loss}'. Choose from {list(lossfuns.keys())}")
            self.lossfunc, self.lossder = lossfuns[loss]
        # Now finding loss for training and validation
        tlosshist = []
        vlosshist = []
        for epoch in range(epochs):
            output = Xtrain
            # Finding output through forward propagation
            for layer in self.layers:
                output = layer.forwardprop(output)
            # if loss function is not specified then selecting loss function appropriately
            if self.lossfunc is None:
                self.selectloss(Ytrain, output)
            # here the loss function for the training data is computed and appended to the the train loss history
            ltrain = self.lossfunc(Ytrain, output)
            tlosshist.append(ltrain)
            # Then error is computed using the loss derivative and back proped
            error = self.lossder(Ytrain, output)
            for layer in reversed(self.layers):
                error = layer.backwardprop(error, lrate)
            # Now validation set is evaluated and the loss function is applied to this too
            valop = self.predict(Xval)
            lval = self.lossfunc(Yval, valop)
            vlosshist.append(lval)
            # Per 100 epochs I print both train and validation loss
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {ltrain}, Validation Loss: {lval}")
        return tlosshist, vlosshist