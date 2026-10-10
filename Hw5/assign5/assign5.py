# Starting code for Assignment 5

#-------------------------------------------------------------------
# 3-layer neural network with 2 inputs, 2 hidden units, and 1 output
#
#                           output
#                             / \
#              output_w1 --->/   \<--- output_w2
#                           /     \
#                       hidden1 hidden2
#                          |\     /|
#           hidden1_w1 --->| \   / |<--- hidden2_w2
#           hidden1_w2 ----|->\ /<-|---- hidden2_w1
#                          |   X   |
#                          |  / \  |
#                        input1 input2
#
#-------------------------------------------------------------------

import random, math
import NNinspector  # automated tester program

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

class NeuralNetwork:

    def __init__(self):
        # unit activations
        self.output = 0
        self.hidden1 = 0
        self.hidden2 = 0
        # output unit weights
        self.output_bias = random.uniform(-0.1, +0.1)
        self.output_w1 = random.uniform(-0.1, +0.1)
        self.output_w2 = random.uniform(-0.1, +0.1)
        # hidden1 unit weights
        self.hidden1_bias = random.uniform(-0.1, +0.1)
        self.hidden1_w1 = random.uniform(-0.1, +0.1)
        self.hidden1_w2 = random.uniform(-0.1, +0.1)
        # hidden2 unit weights
        self.hidden2_bias = random.uniform(-0.1, +0.1)
        self.hidden2_w1 = random.uniform(-0.1, +0.1)
        self.hidden2_w2 = random.uniform(-0.1, +0.1)
        # learning parameters
        self.tolerance = 0.1
        self.epoch_limit = 50000
        self.learning_rate = 0.1
        # for use in Part 2
        self.momentum = 0
        self.output_bias_change = 0
        self.hidden1_bias_change = 0
        self.hidden2_bias_change = 0
        self.output_w1_change = 0
        self.output_w2_change = 0
        self.hidden1_w1_change = 0
        self.hidden1_w2_change = 0
        self.hidden2_w1_change = 0
        self.hidden2_w2_change = 0


    def __str__(self):
        s = "           bias     w1      w2\n"
        s += "output   {:+.4f} {:+.4f} {:+.4f}\n".format(
            self.output_bias, self.output_w1, self.output_w2)
        s += "hidden1  {:+.4f} {:+.4f} {:+.4f}\n".format(
            self.hidden1_bias, self.hidden1_w1, self.hidden1_w2)
        s += "hidden2  {:+.4f} {:+.4f} {:+.4f}".format(
            self.hidden2_bias, self.hidden2_w1, self.hidden2_w2)
        return s
        
    def set_weights(self, weight_list):
        assert type(weight_list) is list and len(weight_list) == 9, \
            "set_weights requires a list of 9 weight/bias values"
        # weights in weight_list must be given in the order below:
        self.output_bias  = weight_list[0]
        self.output_w1    = weight_list[1]
        self.output_w2    = weight_list[2]
        self.hidden1_bias = weight_list[3]
        self.hidden1_w1   = weight_list[4]
        self.hidden1_w2   = weight_list[5]
        self.hidden2_bias = weight_list[6]
        self.hidden2_w1   = weight_list[7]
        self.hidden2_w2   = weight_list[8]
        self.output_bias_change = 0
        self.hidden1_bias_change = 0
        self.hidden2_bias_change = 0
        self.output_w1_change = 0
        self.output_w2_change = 0
        self.hidden1_w1_change = 0
        self.hidden1_w2_change = 0
        self.hidden2_w1_change = 0
        self.hidden2_w2_change = 0

    def initialize(self):
        self.set_weights([random.uniform(-0.1, +0.1) for i in range(9)])
        print("weights randomized")

    def test(self, patterns, targets):
        for pattern, target in zip(patterns, targets):
            output = self.propagate(pattern)
            if abs(target - output) <= self.tolerance:
                print(f"{pattern} --> {output}")
            else:
                print(f"{pattern} --> {output} \t(WRONG, should be {target})")
        error, correct = self.total_error(patterns, targets)
        print(f"TSS error = {error:.5f}, correct = {correct:.3f}")

    #--------------------------------------------------------------------------
    # methods to be implemented

    # returns the output produced by the network for the given pattern
    def propagate(self, pattern):
        x1, x2 = pattern
        self.hidden1 = sigmoid(x1 * self.hidden1_w1 + x2 * self.hidden1_w2 + self.hidden1_bias)
        self.hidden2 = sigmoid(x1 * self.hidden2_w1 + x2* self.hidden2_w2 + self.hidden2_bias)
        self.output = sigmoid(self.hidden1 * self.output_w1 + self.hidden2 * self.output_w2 + self.output_bias)
        return self.output

    # returns a PAIR of values (e, c) where e is the total sum-squared error
    # for all patterns in the given dataset, and c is the fraction of output
    # values that are within self.tolerance of the given target values
    def total_error(self, patterns, targets):
        error = 0
        correct = 0
        for pattern, target in zip(patterns, targets):
            prediction = self.propagate(pattern)
            squared_error = (target - prediction) ** 2
            error += squared_error
            if abs(target - prediction) <= self.tolerance:
                correct += 1
        accuracy = correct / len(patterns)
        return error, accuracy

    # updates network weights and biases for the given pattern and target
    def adjust_weights(self, pattern, target):
        x1, x2 = pattern
        self.propagate(pattern)     
        old_w1 = self.output_w1
        old_w2 = self.output_w2
        d_out = (self.output - target) * self.output * (1-self.output)
        dh1 = (d_out * old_w1) * self.hidden1 * (1-self.hidden1)
        dh2 = (d_out * old_w2) * self.hidden2 * (1-self.hidden2)

        new_output_w1_change = -self.learning_rate * d_out * self.hidden1 + self.momentum * self.output_w1_change
        new_output_w2_change = -self.learning_rate * d_out * self.hidden2 + self.momentum * self.output_w2_change
        new_output_bias_change = -self.learning_rate * d_out + self.momentum * self.output_bias_change

        new_hidden1_w1_change = -self.learning_rate *dh1 * x1  + self.momentum * self.hidden1_w1_change
        new_hidden1_w2_change = -self.learning_rate *dh1 * x2 + self.momentum * self.hidden1_w2_change 
        new_hidden1_bias_change = -self.learning_rate * dh1 + self.momentum * self.hidden1_bias_change

        new_hidden2_w1_change = -self.learning_rate *dh2 * x1  + self.momentum * self.hidden2_w1_change
        new_hidden2_w2_change = -self.learning_rate *dh2 * x2 + self.momentum * self.hidden2_w2_change 
        new_hidden2_bias_change = -self.learning_rate * dh2 + self.momentum * self.hidden2_bias_change

        self.output_w1 += new_output_w1_change
        self.output_w2 += new_output_w2_change
        self.output_bias += new_output_bias_change

        self.hidden1_w1 += new_hidden1_w1_change
        self.hidden1_w2 += new_hidden1_w2_change
        self.hidden1_bias += new_hidden1_bias_change

        self.hidden2_w1 += new_hidden2_w1_change
        self.hidden2_w2 += new_hidden2_w2_change
        self.hidden2_bias += new_hidden2_bias_change

        self.output_w1_change = new_output_w1_change
        self.output_w2_change = new_output_w2_change
        self.output_bias_change = new_output_bias_change

        self.hidden1_w1_change = new_hidden1_w1_change
        self.hidden1_w2_change = new_hidden1_w2_change
        self.hidden1_bias_change = new_hidden1_bias_change

        self.hidden2_w1_change = new_hidden2_w1_change
        self.hidden2_w2_change = new_hidden2_w2_change
        self.hidden2_bias_change = new_hidden2_bias_change

    # trains the network on all of the given patterns and targets
    # until all outputs are within tolerance of the targets
    def train(self, patterns, targets):
        for epoch in range(self.epoch_limit + 1): 
            error, correct = self.total_error(patterns,targets)
            print(f"Epoch #{epoch:4d}: TSS Error = {error:.5f}, Correct = {correct:.3f}")
            if correct == 1.0: 
                return epoch 
            if epoch == self.epoch_limit: 
                print(f"Failed to learn targets within {self.epoch_limit} epochs")
                return 
            for pattern, target in zip(patterns, targets): 
                self.adjust_weights(pattern,target)


def run_experiments(targets, learning_rate, momentum, trials = 5): 
    results = []
    failed = 0
    for trial in range(trials): 
        n = NeuralNetwork()
        n.learning_rate = learning_rate
        n.momentum = momentum 
        n.initialize()
        epochs = n.train(inputs, targets)
        if epochs is not None:
            results.append(epochs)
            print(f"Trial {trial+1}: {epochs} epochs")
        else:
            print(f"Trial {trial+1}: Failed")
            failed += 1

        if len(results) > 0:
            avg = sum(results) / len(results)
            print(f"Average epochs (successful runs only): {avg:.2f}, failed: {failed}")
        else:
            print("No successful runs")
        print(f"Failed trials: {failed} out of {trials}")


#------------------------------------------------------------------------------

inputs = [[0,0], [0,1], [1,0], [1,1]]

ANDtargets  = [0, 0, 0, 1]
ORtargets   = [0, 1, 1, 1]
NANDtargets = [1, 1, 1, 0]
NORtargets  = [1, 0, 0, 0]
XORtargets  = [0, 1, 1, 0]  # not linearly separable
XNORtargets = [1, 0, 0, 1]  # not linearly separable

n = NeuralNetwork()

