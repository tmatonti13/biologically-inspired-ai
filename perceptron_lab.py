# Starting code for Perceptron Lab

import random
#import os
# print(os.listdir())

class Perceptron:

    # constructor
    def __init__(self, num_inputs=2):
        self.num_inputs = num_inputs
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.1, 0.1)
        self.learning_rate = 0.1
        self.updates = 0

    # sets the weights and bias to the given values
    def set_weights(self, weight_list, b): 
        self.weights = list(weight_list)
        self.bias = b

    # prints out the current weight and bias values
    def show_weights(self):
        w_str = " ".join(f"{w:+.4f}" for w in self.weights)
        print(f"{w_str} {self.bias:+.4f}")

    # computes the output of the perceptron given an input pattern
    def propagate(self,pattern):
        total = sum(w * x for w, x in zip(self.weights, pattern)) + self.bias
        if total >= 0:
            output = 1
        else:
            output = 0
        return output

    # returns the total error for the given set of patterns
    def total_error(self, patterns, targets):
        sum_squared_error = 0
        for pattern, target in zip(patterns, targets):
            output = self.propagate(pattern)
            sum_squared_error += (target - output) ** 2
        return sum_squared_error

    # trains the perceptron to completion on the given set of patterns
    def train(self, patterns, targets):
        error = self.total_error(patterns, targets)
        while error > 0:
            # one training epoch
            for pattern, target in zip(patterns, targets):
                self.adjust_weights(pattern, target)
            error = self.total_error(patterns, targets)
        print("All patterns learned")

    # updates the weights and bias for a single pattern/target pair
    def adjust_weights(self, pattern, target):
        output = self.propagate(pattern)
        if output == target:
            # no weight changes necessary
            pass
        else:
            error = target - output
            for i in range(self.num_inputs):
                self.weights[i] += self.learning_rate * pattern[i] * error
            self.bias += self.learning_rate * error
            self.updates +=1
            

            
    # shows the current output of the perceptron on the given patterns
    def test(self, patterns, targets):
        for pattern, target in zip(patterns, targets):
            output = self.propagate(pattern)
            if output == target:
                print(f"{pattern} --> {output}")
            else:
                print(f"{pattern} --> {output}  (WRONG, should be {target})")
        tss_error = self.total_error(patterns, targets)
        print(f"TSS error = {tss_error:.1f}")

#------------------------------------------------------------------
# some input and target patterns

# 2-bit input patterns
patterns = [[0,0], [0,1], [1,0], [1,1]]

# targets for 2-bit input patterns
ANDtargets  = [0, 0, 0, 1]
ORtargets   = [0, 1, 1, 1]
NANDtargets = [1, 1, 1, 0]
NORtargets  = [1, 0, 0, 0]
XORtargets  = [0, 1, 1, 0]

# 3-bit input patterns
patterns3 = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]

# targets for 3-bit input patterns
CARRYtargets = [0, 0, 0, 1, 0, 1, 1, 1]
SUMtargets   = [0, 1, 1, 0, 1, 0, 0, 1]

#------------------------------------------------------------------
# utility functions

# reads in a dataset from a file
def read_dataset(patterns_filename, targets_filename):
    with open(patterns_filename) as file:
        patterns = [[float(x) for x in line.split(',')] for line in file]
    with open(targets_filename) as file:
        targets = [int(line) for line in file]
    return patterns, targets

# randomly reshuffles patterns and targets, but keeps them in sync
def reshuffle_dataset(patterns, targets):
    tuples = list(zip(patterns, targets))
    random.shuffle(tuples)
    shuffled_patterns = [p for (p, t) in tuples]
    shuffled_targets = [t for (p, t) in tuples]
    return shuffled_patterns, shuffled_targets

#------------------------------------------------------------------

p = Perceptron()


### Number 6
##Check for 2 weights 
##p = Perceptron(num_inputs=2)
##p.set_weights([0.1, 0.2], -0.3)
##p.test(patterns, ANDtargets)
##         
##
##
##Carry = Perceptron(num_inputs=3)
##Carry.train(patterns3, CARRYtargets)
##print("Carry updates:", Carry.updates)
##Carry.show_weights()
##Carry.test(patterns3, CARRYtargets)
##
##Sum = Perceptron(num_inputs=3)
##Sum.updates = 0
##Sum.train(patterns3,SUMtargets)
##
##print("Sum updates:", Sum.updates)
##
##
##Sum.show_weights()
##Sum.test(patterns3, SUMtargets)
##
