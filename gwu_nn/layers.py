import numpy as np
from abc import ABC, abstractmethod
from gwu_nn.activation_layers import Sigmoid, RELU, Softmax

activation_functions = {'relu': RELU, 'sigmoid': Sigmoid, 'softmax': Softmax}


def apply_activation_forward(forward_pass):
    """Decorator that ensures that a layer's activation function is applied after the layer during forward
    propagation.
    """
    def wrapper(*args):
        output = forward_pass(args[0], args[1])
        if args[0].activation:
            return args[0].activation.forward_propagation(output)
        else:
            return output
    return wrapper


def apply_activation_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate)
    return wrapper

def apply_adam_activation_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        iteration = args[3]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate, iteration)
    return wrapper


def apply_sgd_activation_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        momentum = args[3]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate, momentum)
    return wrapper

def apply_rms_prop_activation_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate)
    return wrapper

def apply_adadelta_activavtion_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate)
    return wrapper




class Layer():
    """The Layer layer is an abstract object used to define the template
    for other layer types to inherit"""

    def __init__(self, activation=None):
        """Because Layer is an abstract object, we don't provide any detailing
        on the initializtion"""
        self.type = "Layer"
        if activation:
            self.activation = activation_functions[activation]()
        else:
            self.activation = None

    @apply_activation_forward
    def forward_propagation(cls, input):
        """:noindex:"""
        pass

    @apply_activation_backward
    def backward_propogation(cls, output_error, learning_rate):
        """:noindex:"""
        pass

    @apply_adam_activation_backward
    def adam_backward_propogation(cls, output_error, learning_rate, iteration):
        """:noindex:"""
        pass

    @apply_activation_backward
    def sgd_backward_propogation(cls, output_error, learning_rate, momentum):
        """:noindex:"""
        pass


    @apply_rms_prop_activation_backward
    def rms_prop_backward_propogation(cls, output_error, learning_rate):
        """:noindex:"""
        pass


    @apply_adadelta_activavtion_backward
    def adadelta_backward_propogation(cls, output_error, learning_rate):
        """:noindex:"""
        pass

class Dense(Layer):
    """The Dense layer class creates a layer that is fully connected with the previous
    layer. This means that the number of weights will be MxN where M is number of
    nodes in the previous layer and N = number of nodes in the current layer.
    """

    def __init__(self, output_size, add_bias=False, activation=None, input_size=None, B1=0.9, B2=0.999, e=1e-8, decay = 0.9):
        super().__init__(activation)
        self.type = None
        self.name = "Dense"
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = add_bias

        #Here, we start setting our original variables for adam
        #First, we handle mean and variance of the weights:
        self.weightMean, self.weightVariance = 0, 0

        #Then, we handle mean and variance of the bias
        self.biasMean, self.biasVariance = 0, 0

        #Let's instantiate the decay rates for our B1 and B2
        self.B1, self.B2 = B1, B2

        #Define an epsilon value to ensure we don't divide by 0
        self.e = e

        #Define a decay paramater for RMSprop
        self.decay = decay


        # Define variable to track previous running average of squared gradient, start at 0
        self.eg = 0

        # Define variable to track previous running average of sqaured paramter updates, start at 0
        self.e_theta = 0

        # Define variable to track current update vector for SGD with momentum
        self.vt = 0

    def init_weights(self, input_size):
        """Initialize the weights for the layer based on input and output size

        Args:
            input_size (numpy array): dimensions for the input array
        """
        if self.input_size is None:
            self.input_size = input_size

        self.weights = np.random.randn(input_size, self.output_size) / np.sqrt(input_size + self.output_size)

        # TODO: Batching of inputs has broken how bias works. Need to address in next iteration
        if self.add_bias:
            self.bias = np.random.randn(1, self.output_size) / np.sqrt(input_size + self.output_size)

    def init_weights_seeded(self, input_size, random_seed):
        """Initialize the weights for the layer based on input and output size

        Args:
            input_size (numpy array): dimensions for the input array
        """

        np.random.seed(seed=random_seed)

        if self.input_size is None:
            self.input_size = input_size

        self.weights = np.random.randn(input_size, self.output_size) / np.sqrt(input_size + self.output_size)

        # TODO: Batching of inputs has broken how bias works. Need to address in next iteration
        if self.add_bias:
            self.bias = np.random.randn(1, self.output_size) / np.sqrt(input_size + self.output_size)


    @apply_activation_forward
    def forward_propagation(self, input):
        """Applies the forward propagation for a densely connected layer. This will compute the dot product between the
        input value (calculated during forward propagation) and the layer's weight tensor.

        Args:
            input (np.array): Input tensor calculated during forward propagation up to this layer.

        Returns:
            np.array(float): The dot product of the input and the layer's weight tensor."""
        self.input = input
        output = np.dot(input, self.weights)
        if self.add_bias:
            return output + self.bias
        else:
            return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights (dot product of the layer's input values and the output_error).

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""
        input_error = np.dot(output_error, self.weights.T)

        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        if self.add_bias:
            self.bias -= learning_rate * output_error
        return input_error

    @apply_adam_activation_backward
    def adam_backward_propagation(self, output_error, learning_rate, iteration):
        """Applies the adam optimizer backward propagation for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights (dot product of the layer's input values and the output_error).

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""

        #Calculates error
        input_error = np.dot(output_error, self.weights.T)

        #Calculates update gradient
        weights_error = np.dot(self.input.T, output_error)

        #let's calculate the momentum for B1 weights
        self.weightMean = self.B1 * self.weightMean + (1-self.B1) * weights_error

        #time to calculate the updated variances weights
        self.weightVariance = self.B2 * self.weightVariance + (1-self.B2)*(weights_error**2)

        #correct for bias
        meanWeightCorrection = self.weightMean/(1-self.B1**(iteration+1))
        varianceWeightCorrection = self.weightVariance/(1-self.B2**(iteration+1))

        #update weights in the ADAM way:
        self.weights -= learning_rate * (meanWeightCorrection/(np.sqrt(varianceWeightCorrection) + self.e))

        if self.add_bias:
            self.bias -= learning_rate * output_error

        return input_error

    @apply_sgd_activation_backward
    def sgd_backward_propagation(self, output_error, learning_rate, momentum = True):
        """Applies the backward propagation using SGD for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights.

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T[:1,:], output_error)

        # Set tentative update
        self.theta = learning_rate * weights_error

        # If using momentum, change update using update vector
        if momentum:
            self.vt = self.decay * self.vt + self.theta
            self.theta = self.vt

        self.weights -= self.theta
        if self.add_bias:
            self.bias -= learning_rate * output_error
        return input_error

    @apply_rms_prop_activation_backward
    def rms_prop_backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation using RMSprop for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights.

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T[:1,:], output_error)

        # Update running average of squared gradients with current squared gradient
        self.eg = self.decay * self.eg + (1 - self.decay) * np.power(weights_error,2)

        # Update weights according to update equation for RMSprop
        self.weights -=  learning_rate / (np.sqrt(self.eg + self.e)) * weights_error

        if self.add_bias:
            self.bias -= learning_rate * output_error
        return input_error


    @apply_adadelta_activavtion_backward
    def adadelta_backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation using Adadelta for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights.

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T[:1,:], output_error)

        # Updating running average of squared gradients using current squared gradient
        self.eg = self.decay * self.eg + (1 - self.decay) * np.power(weights_error,2)

        # Calculate root mean squared (RMS) error of gradient AFTER running average has been updated
        rms_eg = np.sqrt(self.eg + self.e)

        # Calculate RMS error of paramater updates BEFORE new running average has been updated (RMS at time step t - 1 for future weights update)
        rms_e_theta = np.sqrt(self.e_theta + self.e)

        # Caclulate theta to update runninga verage of squared paramater updates
        delta_theta = - learning_rate / (rms_eg) * weights_error

        # Update running average of squared paramater updates using current squared paramater update
        self.e_theta = self.decay * self.e_theta + (1 - self.decay) * np.power(delta_theta,2)

        # Calculate paramater update for weights
        delta_theta = (rms_e_theta / rms_eg) * weights_error

        # Update weights
        self.weights -= delta_theta

        if self.add_bias:
            self.bias -= learning_rate * output_error

        return input_error