# Starting code for Perceptron Lab

import random

class Perceptron:

    # constructor
    def __init__(self, num_inputs = 2):
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
            self.show_weights()

            
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


# number 1:
# The ANDtargets produces a TSS error of 0, being the only of each to do so.
# The ORtargets produces a TSS error of 2, the NAND produces one of 4,
# the NOR produces one of 2,and the XOR produces one of 3. The weights are
# most effective with the AND function, and least effective with the NAND function


# number 2:
    ##targets = [1,0,1,1]
    ##p.set_weights(0.1, 0.2, 0.3)
    ##print(p.total_error(patterns, targets))
    ##1
    ##p.set_weights(0.0,0.0,0.0)
    ##print(p.total_error(patterns, targets))
    ##1
    ##p.set_weights(-0.1, -0.2, -0.3)
    ##print(p.total_error(patterns, targets))
    ##3
    ##p.set_weights(1.2, -2.3, -0.2)
    ##print(p.total_error(patterns, targets))
    ##2
    ##p.set_weights(-0.3,0.3,-0.3)
    ##print(p.total_error(patterns, targets))
    ##4
    ##p.set_weights(0.2,-0.1,0.1)
    ##print(p.total_error(patterns, targets))
    ##1


    ##| weight1 | weight2 | bias | TSS error |
    ##| ------- | ------- | ---- | --------- |
    ##| 0.1     | 0.2     | 0.3  | **1**     |
    ##| 0.0     | 0.0     | 0.0  | **1**     |
    ##| -0.1    | -0.2    | -0.3 | **3**     |
    ##| 1.2     | -2.3    | -0.2 | **2**     |
    ##| -0.3    | 0.3     | -0.3 | **4**     |
    ##| 0.2     | -0.1    | 0.1  | **1**     |


# number 3:

##targets = [1,0,1,1]
##p.set_weights(-0.3, 0.3, -0.3)
##p.train(patterns, targets)
##pattern: [0, 0], new weights: -0.3000 +0.3000 -0.2000
##pattern: [0, 1], new weights: -0.3000 +0.2000 -0.3000
##pattern: [1, 0], new weights: -0.2000 +0.2000 -0.2000
##pattern: [1, 1], new weights: -0.1000 +0.3000 -0.1000
##pattern: [0, 0], new weights: -0.1000 +0.3000 +0.0000
##pattern: [0, 1], new weights: -0.1000 +0.2000 -0.1000
##pattern: [1, 0], new weights: +0.0000 +0.2000 +0.0000
##pattern: [0, 1], new weights: +0.0000 +0.1000 -0.1000
##pattern: [1, 0], new weights: +0.1000 +0.1000 +0.0000
##pattern: [0, 1], new weights: +0.1000 -0.0000 -0.1000
##pattern: [0, 0], new weights: +0.1000 -0.0000 +0.0000
##pattern: [0, 1], new weights: +0.1000 -0.1000 -0.1000
##pattern: [1, 1], new weights: +0.2000 -0.0000 +0.0000
##pattern: [0, 1], new weights: +0.2000 -0.1000 -0.1000
##pattern: [0, 0], new weights: +0.2000 -0.1000 +0.0000
##All patterns learned


## Starting from the maximum-error weights (-0.3, 0.3) with bias -0.3, the perceptron took 15 weight updates to learn the targets [1,0,1,1].
## The final learned parameters were w1 = 0.2, w2 = -0.1, bias = 0.0

# number 4:

##targets = [0,1,0,0]
##p.set_weights(0.2, -0.1, 0.0)
##p.train(patterns, targets)
##pattern: [0, 0], new weights: +0.2000 -0.1000 -0.1000
##pattern: [0, 1], new weights: +0.2000 +0.0000 +0.0000
##pattern: [1, 0], new weights: +0.1000 +0.0000 -0.1000
##pattern: [1, 1], new weights: +0.0000 -0.1000 -0.2000
##pattern: [0, 1], new weights: +0.0000 +0.0000 -0.1000
##pattern: [0, 1], new weights: +0.0000 +0.1000 +0.0000
##pattern: [1, 0], new weights: -0.1000 +0.1000 -0.1000
##All patterns learned

# It took 7 weight updates for the perceptron to learn correctly. The final learned parameters were w1 = -0.1, w2 = 0.1, bias = -0.1.
# The perceptron can be retrained to represent other logical functions such as AND, OR, NAND,
# and NOR because they are linearly separable, but it cannot learn XOR because XOR is not linearly separable.

# number 5:

##Carry = Perceptron()         
##Carry.updates = 0
##Carry.train(patterns3, CARRYtargets)

##pattern: [0, 0, 0], new weights: -0.0507 -0.0389 +0.0080 -0.0403
##pattern: [0, 1, 1], new weights: -0.0507 +0.0611 +0.1080 +0.0597
##pattern: [1, 0, 0], new weights: -0.1507 +0.0611 +0.1080 -0.0403
##pattern: [1, 0, 1], new weights: -0.0507 +0.0611 +0.2080 +0.0597
##pattern: [0, 0, 0], new weights: -0.0507 +0.0611 +0.2080 -0.0403
##pattern: [0, 0, 1], new weights: -0.0507 +0.0611 +0.1080 -0.1403
##pattern: [1, 0, 1], new weights: +0.0493 +0.0611 +0.2080 -0.0403
##pattern: [0, 0, 1], new weights: +0.0493 +0.0611 +0.1080 -0.1403
##pattern: [1, 1, 0], new weights: +0.1493 +0.1611 +0.1080 -0.0403
##pattern: [0, 0, 1], new weights: +0.1493 +0.1611 +0.0080 -0.1403
##pattern: [0, 1, 0], new weights: +0.1493 +0.0611 +0.0080 -0.2403
##pattern: [0, 1, 1], new weights: +0.1493 +0.1611 +0.1080 -0.1403
##pattern: [1, 0, 0], new weights: +0.0493 +0.1611 +0.1080 -0.2403
##pattern: [1, 0, 1], new weights: +0.1493 +0.1611 +0.2080 -0.1403
##pattern: [0, 0, 1], new weights: +0.1493 +0.1611 +0.1080 -0.2403
##All patterns learned
##print("Carry updates:", Carry.updates)
       
##Carry updates: 15
##Carry.show_weights()         
##+0.1493 +0.1611 +0.1080 -0.2403

##Sum.train(patterns3, SUMtargets)
##          
##pattern: [0, 0, 1], new weights: +0.1422 -0.0336 +0.0811 +0.0429
##pattern: [0, 1, 1], new weights: +0.1422 -0.1336 -0.0189 -0.0571
##pattern: [1, 0, 1], new weights: +0.0422 -0.1336 -0.1189 -0.1571
##pattern: [1, 1, 1], new weights: +0.1422 -0.0336 -0.0189 -0.0571
##pattern: [0, 0, 1], new weights: +0.1422 -0.0336 +0.0811 +0.0429
##pattern: [0, 1, 1], new weights: +0.1422 -0.1336 -0.0189 -0.0571
##pattern: [1, 0, 1], new weights: +0.0422 -0.1336 -0.1189 -0.1571
# ... etc
##Traceback (most recent call last):
##  File "<pyshell#48>", line 1, in <module>
##    Sum.train(patterns3, SUMtargets)
##  File "C:/Users/thoma/OneDrive/Desktop/Bio AI/perceptron_lab.py", line 51, in train
##    self.adjust_weights(pattern, target)
##  File "C:/Users/thoma/OneDrive/Desktop/Bio AI/perceptron_lab.py", line 75, in adjust_weights
##    print(f"pattern: {pattern}, new weights: {self.weight1:+.4f} " +
##KeyboardInterrupt
##Sum.test(patterns3, SUMtargets)
##          
##[0, 0, 0] --> 0
##[0, 0, 1] --> 0  (WRONG, should be 1)
##[0, 1, 0] --> 0  (WRONG, should be 1)
##[0, 1, 1] --> 0
##[1, 0, 0] --> 1
##[1, 0, 1] --> 1  (WRONG, should be 0)
##[1, 1, 0] --> 0
##[1, 1, 1] --> 0  (WRONG, should be 1)
##TSS error = 4.0
##print("Sum updates:", Sum.updates)
##          
##Sum updates: 149 (did not ever converge this is just until i executed it)

# For the carry perceptron, yes it can learn a correct set of weights as it eventually reached a TSS error of
# 0. In my run it took 15 weight updates to converge and the final parameters were w1 = 0.1493, w2 = 0.1611, w3 = 0.1080, bias = −0.2403.


# For the sum perceptron, no it cannot learn the task to completion becasue the SUM output is an XOR like pattern. If we view the 8 input patterns
# as the vertices of a cube, the 0 or 1 values for the sum would alternate so there would be no single plane that can seperate all of the 1
# labeled vertices from the 0's. Since the perceptron only makes a plane in 3D, the training does not converge to an error of 0.




# number 6:

# Check for 2 weights 
##p = Perceptron(num_inputs=2)
##          
##p.set_weights([0.1, 0.2], -0.3)
##          
##p.test(patterns, ANDtargets)
##          
##[0, 0] --> 0
##[0, 1] --> 0
##[1, 0] --> 0
##[1, 1] --> 1
##TSS error = 0.0

##Carry = Perceptron(num_inputs=3)
##          
##Carry.updates = 0
##
##Carry.train(patterns3, CARRYtargets)
##
##+0.0945 +0.0080 +0.1040 +0.0002
##-0.0055 +0.0080 +0.1040 -0.0998
##+0.0945 +0.0080 +0.2040 +0.0002
##+0.0945 +0.0080 +0.2040 -0.0998
##+0.0945 +0.0080 +0.1040 -0.1998
##+0.0945 +0.1080 +0.2040 -0.0998
##+0.0945 +0.1080 +0.1040 -0.1998
##+0.1945 +0.1080 +0.2040 -0.0998
##+0.1945 +0.1080 +0.1040 -0.1998
##All patterns learned
##print("Carry updates:", Carry.updates)
##
##Carry updates: 9
##Carry.show_weights()
##
##+0.1945 +0.1080 +0.1040 -0.1998
##Carry.test(patterns3, CARRYtargets)
##
##[0, 0, 0] --> 0
##[0, 0, 1] --> 0
##[0, 1, 0] --> 0
##[0, 1, 1] --> 1
##[1, 0, 0] --> 0
##[1, 0, 1] --> 1
##[1, 1, 0] --> 1
##[1, 1, 1] --> 1
##TSS error = 0.0
##Sum = Perceptron(num_inputs=3)
##
##Sum.updates = 0
##
##Sum.train(patterns3,SUMtargets)
##          
##-0.0741 -0.0540 -0.0674 -0.0824
##-0.0741 -0.0540 +0.0326 +0.0176
##-0.0741 +0.0460 +0.0326 +0.1176
##-0.0741 -0.0540 -0.0674 +0.0176
##+0.0259 -0.0540 -0.0674 +0.1176
##-0.0741 -0.0540 -0.1674 +0.0176
##+0.0259 +0.0460 -0.0674 +0.1176
##+0.0259 +0.0460 -0.0674 +0.0176
##+0.0259 +0.0460 +0.0326 +0.1176
##+0.0259 -0.0540 -0.0674 +0.0176
##+0.1259 +0.0460 +0.0326 +0.1176
##+0.1259 +0.0460 +0.0326 +0.0176
##+0.1259 -0.0540 -0.0674 -0.0824
##+0.2259 +0.0460 +0.0326 +0.0176
##+0.2259 +0.0460 +0.0326 -0.0824
##+0.2259 +0.0460 +0.1326 +0.0176
##+0.2259 -0.0540 +0.0326 -0.0824
##+0.1259 -0.0540 -0.0674 -0.1824
##+0.2259 +0.0460 +0.0326 -0.0824
##+0.2259 +0.0460 +0.1326 +0.0176
##+0.2259 -0.0540 +0.0326 -0.0824
##+0.1259 -0.0540 -0.0674 -0.1824
##+0.2259 +0.0460 +0.0326 -0.0824
##+0.2259 +0.0460 +0.1326 +0.0176
##+0.2259 -0.0540 +0.0326 -0.0824
##+0.1259 -0.0540 -0.0674 -0.1824
##+0.2259 +0.0460 +0.0326 -0.0824
##+0.2259 +0.0460 +0.1326 +0.0176
##+0.2259 -0.0540 +0.0326 -0.0824
##+0.1259 -0.0540 -0.0674 -0.1824
##+0.2259 +0.0460 +0.0326 -0.0824
##Traceback (most recent call last):
##  File "<pyshell#64>", line 1, in <module>
##    Sum.train(patterns3,SUMtargets)
##  File "C:/Users/thoma/OneDrive/Desktop/Bio AI/perceptron_lab.py", line 48, in train
##    self.adjust_weights(pattern, target)
##  File "C:/Users/thoma/OneDrive/Desktop/Bio AI/perceptron_lab.py", line 64, in adjust_weights
##    self.show_weights()
##  File "C:/Users/thoma/OneDrive/Desktop/Bio AI/perceptron_lab.py", line 23, in show_weights
##    print(f"{w_str} {self.bias:+.4f}")
##KeyboardInterrupt
##print("Sum updates:", Sum.updates)
##
##Sum updates: 31
##Sum.show_weights()
##
##+0.2259 +0.0460 +0.0326 -0.0824
##Sum.test(patterns3, SUMtargets)
##          
##[0, 0, 0] --> 0
##[0, 0, 1] --> 0  (WRONG, should be 1)
##[0, 1, 0] --> 0  (WRONG, should be 1)
##[0, 1, 1] --> 0
##[1, 0, 0] --> 1
##[1, 0, 1] --> 1  (WRONG, should be 0)
##[1, 1, 0] --> 1  (WRONG, should be 0)
##[1, 1, 1] --> 1
##TSS error = 4.0


# I generalized the perceptron by instead of having a set number of weights, I put them in a list. Then propagate() now computes the dot product
# between the weights list and the input pattern. Then adjust_weights() loops over the inputs to apply the perceptron learning. The class still works for
# the 2 and 3 input.


# number 7

samples, diagnosis = read_dataset("cancerPatterns.txt", "cancerTargets.txt")
training_samples, training_targets = samples[0:275], diagnosis[0:275]
testing_samples, testing_targets = samples[275:550], diagnosis[275:550]

for i in range(5):
    print("Trial", i+1)

    p = Perceptron(num_inputs=30)
    p.updates = 0

    p.train(training_samples, training_targets)

    print("Weight updates:", p.updates)

    correct = 0
    for pat, targ in zip(testing_samples, testing_targets):
        if p.propagate(pat) == targ:
            correct += 1

    accuracy = correct / len(testing_targets) * 100
    print("Accuracy:", accuracy, "%")
    print()







