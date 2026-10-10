# Automated tester program for Backpropagation Assignment

import random, math, sys
#------------------------------------------------------------------

inputs = [[0,0], [0,1], [1,0], [1,1]]

XORtargets  = [0, 1, 1, 0]

# order matters
_activation_vars = ['hidden1', 'hidden2', 'output']

# order matters
_weight_vars = ['output_bias', 'output_w1', 'output_w2',
                'hidden1_bias', 'hidden1_w1', 'hidden1_w2',
                'hidden2_bias', 'hidden2_w1', 'hidden2_w2']

_parameter_vars = ['tolerance', 'learning_rate', 'momentum'] #, 'sigmoid_prime_offset']

def _test_constructor(NeuralNetwork):
    n = NeuralNetwork()
    attributes = dir(n)
    for var in _activation_vars + _weight_vars + _parameter_vars:
        if var not in attributes:
            raise Exception("__init__ did not initialize self.%s" % (var,))
    for var in _activation_vars:
        value = getattr(n, var)
        if value == 0:
            pass
        else:
            raise Exception("__init__ should set self.%s to 0 instead of %s" % (var, value))
    for var in _weight_vars:
        value = getattr(n, var)
        if type(value) in (int, float) and -0.1 <= value <= +0.1 and value != 0:
            pass
        else:
            raise Exception("__init__ should set self.%s to a random value between -0.1 and +0.1 instead of %s" % (var, value))
    for var in _parameter_vars:
        value = getattr(n, var)
        if type(value) in (int, float) and 0 <= value <= 1:
            pass
        else:
            raise Exception("__init__ should set self.%s to a number in the range 0 to 1 instead of %s" % (var, value))
    print("OK")

_test_weights = (
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
)

def _test_set_weights(NeuralNetwork):
    n = NeuralNetwork()
    for weights in _test_weights:
        n.set_weights(weights)
        attributes = dir(n)
        for (w, var) in zip(weights, _weight_vars):
            if var not in attributes:
                raise Exception("set_weights method did not initialize self.%s" % (var,))
            if getattr(n, var) != w:
                raise Exception("set_weights method did not set self.%s correctly" % (var,))
    print("OK")

def approx_equal(x1, x2):
    test_tolerance = 1e-9
    return abs(x1 - x2) <= test_tolerance

def _test_propagate(NeuralNetwork, trace):
    n = NeuralNetwork()
    for weights in _test_weights:
        n.set_weights(weights)
        for pattern in inputs:
            result = n.propagate(pattern)
            correct = trace.next()
            if type(result) not in (int, float) or not approx_equal(result, correct):
                print("\nProblem detected in propagate method:")
                print("propagating input %s with weights %s" % (pattern, weights), end=" ")
                raise Exception("propagate method returned %s but should have returned %s" % (result, correct))
            for var in _activation_vars:
                value = getattr(n, var)
                correct = trace.next()
                if not approx_equal(value, correct):
                    print("\nProblem detected in propagate method:")
                    print("propagating input %s with weights %s" % (pattern, weights))
                    print("self.%s activation should be %s but is %s" % (var, correct, value), end=" ")
                    raise Exception("propagate method did not update self.%s activation correctly" % (var,))
    print("OK")

def _test_total_error(NeuralNetwork, trace):
    n = NeuralNetwork()
    n.tolerance = 0.1
    test_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_targets = trace.next()
    test_cases = (
        # basic tests
        {'targets': [0,0,0,0], 'weights': test_weights},
        {'targets': [1,1,1,1], 'weights': test_weights},
        {'targets': [0,1,0,1], 'weights': test_weights},
        # for testing tolerance
        {'targets': test_targets, 'weights': test_weights},
        {'targets': [t+0.09 for t in test_targets], 'weights': test_weights},
        {'targets': [t-0.09 for t in test_targets], 'weights': test_weights},
        {'targets': [t+0.11 for t in test_targets], 'weights': test_weights},
        {'targets': [t-0.11 for t in test_targets], 'weights': test_weights},
        # XOR targets
        {'targets': XORtargets,
         'weights': # 0 patterns correct
         [0.012963972353188247, 0.03645275632478018, 0.07302358512367513,
          0.06809033551217336, 0.08705535741734996, -0.07920038721617007,
          -0.0157202269102054, -0.05114617857796531, -0.047243258620952605]},
        {'targets': XORtargets,
         'weights': # 1 pattern correct
         [2.774999285987921, -4.517732789764918, -5.164517867241201,
          3.302537131754974, -2.941763517083835, -3.0823933468937152,
          3.8897306177263586, -3.4062297670291937, -3.2829376308924942]},
        {'targets': XORtargets,
         'weights': # 2 patterns correct
         [9.458704474085394, -6.965332636379139, -5.306377679490277,
          -4.365487225458641, 3.060880707916936, 6.772884024179794,
          6.54734584564512, -3.635241367757549, -0.7420732776300978]},
        {'targets': XORtargets,
         'weights': # 3 patterns correct
         [7.202016691883634, -4.253312909977837, -7.598120967599434,
          -0.102395629329159, 5.902878477068958, -1.8606788462700057,
          7.536765490270358, 0.6307730661491615, -7.182416854751768]},
        {'targets': XORtargets,
         'weights': # 4 patterns correct
         [9.51782505128544, -6.194149676267264, -6.812448178305661,
          1.8701450492967449, 4.991767478048334, -4.299700250392367,
          5.120372726507042, -7.0323938776683015, 4.092735535823893]}
    )
    for case in test_cases:
        targets = case['targets']
        weights = case['weights']
        n.set_weights(weights)
        result = n.total_error(inputs, targets)
        if type(result) not in (list, tuple) or len(result) != 2 or \
           type(result[0]) not in (int, float) or type(result[1]) not in (int, float):
            print("\nProblem detected in total_error method:")
            print("total_error should return the TSS error and the fraction of correct outputs", end=" ")
            raise Exception("total_error method did not return a tuple of two numbers")
        value1, value2 = result
        error, correct = trace.next()
        if not approx_equal(value1, error):
            print("\nProblem detected in total_error method:")
            print("testing with targets %s and weights %s" % (targets, weights))
            print("returned %s for the TSS error but should have returned %s" % (value1, error), end=" ")
            raise Exception("total_error method did not correctly compute the TSS error")
        if not (0 <= value2 <= 1):
            print("\nProblem detected in total_error method:")
            print("testing with targets %s and weights %s" % (targets, weights))
            print("returned %s for the fraction of correct outputs, but should be in the range 0-1" % (value2,), end=" ")
            raise Exception("total_error method did not correctly compute the fraction of correct outputs")
        if not approx_equal(value2, correct):
            print("\nProblem detected in total_error method:")
            print("testing with targets %s and weights %s" % (targets, weights))
            print("returned %s for the fraction of correct outputs but should be %s" % (value2, correct), end=" ")
            raise Exception("total_error method did not correctly compute the fraction of correct outputs")
    print("OK")

def _test_adjust_weights(NeuralNetwork, trace, momentum=0):
    n = NeuralNetwork()
    n.learning_rate = 0.1
    n.momentum = momentum
    n.sigmoid_prime_offset = 0
    n.tolerance = 0.1
    for weights in _test_weights:
        n.set_weights(weights)
        for (pattern, target) in zip(inputs, XORtargets):
            n.adjust_weights(pattern, target)
            for var in [#'output', 'hidden1', 'hidden2',
                        'hidden1_w1', 'hidden1_w2', 'hidden2_w1',
                        'hidden2_w2', 'output_w1', 'output_w2',
                        'hidden1_bias', 'hidden2_bias', 'output_bias']:
                value = getattr(n, var)
                correct = trace.next()
                if not approx_equal(value, correct):
                    print("\nProblem detected in adjust_weights method:")
                    print("testing with input %s, target %s, learning rate %s, momentum %s" %
                        (pattern, target, n.learning_rate, n.momentum))
                    print("and weights %s" % weights)
                    print("self.%s set to %s but should be %s" % (var, value, correct), end=" ")
                    raise Exception("adjust_weights method did not update self.%s correctly" % (var,))
    print("OK")

def _test_train(NeuralNetwork, momentum=0):
    n = NeuralNetwork()
    n.learning_rate = 0.3
    n.momentum = momentum
    n.sigmoid_prime_offset = 0
    n.tolerance = 0.1
    # second set of weights takes longer to learn, so test that one first
    num = 0
    for weights in _test_weights[::-1]:
        momentum_desc = 'no momentum' if n.momentum == 0 else ('momentum %s' % n.momentum)
        print("\nHit RETURN to train on XOR with learning rate %s, %s,\nand weights %s ..." %
              (n.learning_rate, momentum_desc, weights), end=" ")
        answer = input()
        if answer in ('q', 'Q'):
            raise KeyboardInterrupt
        n.set_weights(weights)
        n.train(inputs, XORtargets)
        error, correct = n.total_error(inputs, XORtargets)
        if correct == 1.0:
            pass
        else:
            print("\nTesting trained XOR network...")
            n.test(inputs, XORtargets)
            raise Exception("train method failed to learn all XOR patterns")
        if momentum == 0:
            print("\nNetwork successfully learned XOR")
        else:
            print("\nNetwork successfully learned XOR with momentum")

_exception_occurred = False

def test():
    print("Use NNinspector.part1() or NNinspector.part2()")

# def test():
#     global _exception_occurred
#     _exception_occurred = False
#     part1()
#     if _exception_occurred: return
#     part2()

def part1():
    global _exception_occurred
    trace = Trace(part1_trace_data)
    main = sys.modules['__main__']
    if 'NeuralNetwork' not in dir(main):
        print("You first need to define a class called NeuralNetwork")
        return
    nnclass = main.__getattribute__('NeuralNetwork')
    try:
        print("Testing __init__ method ...", end=" ")
        _test_constructor(nnclass)
        print("Testing set_weights method ...", end=" ")
        _test_set_weights(nnclass)
        print("Testing propagate method ...", end=" ")
        _test_propagate(nnclass, trace)
        print("Testing total_error method ...", end=" ")
        _test_total_error(nnclass, trace)
        print("Testing adjust_weights method ...", end=" ")
        _test_adjust_weights(nnclass, trace)
        _test_train(nnclass)
        _exception_occurred = False
        return
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("\n*** ERROR: %s" % (e,))
    _exception_occurred = True

def part2():
    trace = Trace(part2_trace_data)
    main = sys.modules['__main__']
    if 'NeuralNetwork' not in dir(main):
        print("You first need to define a class called NeuralNetwork")
        return
    nnclass = main.__getattribute__('NeuralNetwork')
    try:
        print("\nTesting adjust_weights method with momentum ...", end=" ")
        _test_adjust_weights(nnclass, trace, momentum=0.9)
        _test_train(nnclass, momentum=0.9)
        return
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("\n*** ERROR: %s" % (e,))

#------------------------------------------------------------------

class Trace:

    def __init__(self, data):
        self.data = data
        self.n = 0

    def next(self):
        x = self.data[self.n]
        self.n += 1
        return x

    def reset(self):
        self.n = 0


part1_trace_data = [0.603529638808172, 0.598687660112452, 0.6681877721681662, 0.603529638808172,
                    0.621475862704675, 0.7310585786300049, 0.8320183851339245, 0.621475862704675,
                    0.6195084197829274, 0.7109495026250039, 0.8175744761936437, 0.6195084197829274,
                    0.6314780917204893, 0.8175744761936437, 0.9168273035060777, 0.6314780917204893,
                    0.8604022340272325, 0.6456563062257954, 0.574442516811659, 0.8604022340272325,
                    0.870335477518413, 0.7310585786300049, 0.598687660112452, 0.870335477518413,
                    0.8739042874039081, 0.7502601055951177, 0.6224593312018546, 0.8739042874039081,
                    0.8814275868365138, 0.8175744761936437, 0.6456563062257954, 0.8814275868365138,
                    [0.6, 0.62, 0.62, 0.63], (1.533035535349333, 0.0), (0.5810515093168057, 0.0),
                    (1.0271276264990044, 0.0), (1.706292708320096e-05, 1.0), (0.03133850058415571, 1.0),
                    (0.03349562527001065, 1.0), (0.04709882006350517, 0.0), (0.04973530579066121, 0.0),
                    (1.0011282905237373, 0.0), (2.6382018304585113, 0.25), (1.792488636574033, 0.5),
                    (1.0030950773944391, 0.75), (0.02826544357070063, 1.0), 0.5, 0.6, 0.8,
                    0.9, 0.19135413920546757, 0.29035046344250753, 0.39930606187484086, 0.6990394497331941,
                    0.08555864539965884, 0.5, 0.6003434815213188, 0.8, 0.9003706043971074, 0.1980250866609567,
                    0.29794286612706683, 0.3996495433961596, 0.6994100541303014, 0.09468540086480275,
                    0.5003676804255118, 0.6003434815213188, 0.8004015894048657, 0.9003706043971074,
                    0.20444705817115721, 0.3053279289387997, 0.4000172238216715, 0.6998116435351671,
                    0.1037192663418811, 0.49991923179384756, 0.5998950328896545, 0.8000591818600866,
                    0.9000281968523283, 0.19241597084156023, 0.29183741528265617, 0.3995687751900072,
                    0.699469235990388, 0.0890056352081807, 0.5, 0.4, 0.2, 0.1, 0.7933275869342599,
                    0.6940635323813434, 0.5981085380063187, 0.29823158424682283, 0.8896656890029558,
                    0.5, 0.40023565997048643, 0.2, 0.10025181256395989, 0.7944305861158023,
                    0.6949666327467173, 0.5983441979768052, 0.29848339681078273, 0.8911752271332227,
                    0.5002126537273965, 0.40023565997048643, 0.2002332154505678, 0.10025181256395989,
                    0.7955010936162923, 0.6958546466347423, 0.5985568517042017, 0.2987166122613505,
                    0.8926026661443992, 0.49910911357323706, 0.39913211981632696, 0.19875306569343168,
                    0.09877166280682378, 0.7879029315023566, 0.6898548215946203, 0.5974533115500422,
                    0.29723646250421437, 0.8833074374400998]

part2_trace_data = [0.5, 0.6, 0.8, 0.9, 0.19135413920546757, 0.29035046344250753, 0.39930606187484086,
                    0.6990394497331941, 0.08555864539965884, 0.5, 0.6003434815213188, 0.8, 0.9003706043971074,
                    0.19024381194587747, 0.2892582832253236, 0.39902499908351635, 0.6985455588901761,
                    0.0816881817244957, 0.5003610649914836, 0.6006526148905057, 0.8003986424956303,
                    0.900704148354504, 0.19580597190372323, 0.29582095665783525, 0.39913310756280784,
                    0.6984996996270902, 0.08743650949528593, 0.5002578601475929, 0.600502671586548,
                    0.8004265546372208, 0.9006734718116841, 0.18882382404179007, 0.2882847315331726,
                    0.39880224185794433, 0.6981275601858362, 0.07794740043007259, 0.5,
                    0.4, 0.2, 0.1, 0.7933275869342599, 0.6940635323813434, 0.5981085380063187,
                    0.29823158424682283, 0.8896656890029558, 0.5, 0.40023565997048643, 0.2,
                    0.10025181256395989, 0.7884254143566362, 0.6896238118899265, 0.5966418821824919,
                    0.29689182263292324, 0.881874347235883, 0.500217322381709, 0.40044775394392423,
                    0.20023819443398022, 0.10047844387152378, 0.7851143953276908, 0.6865411582629664,
                    0.5955392143227568, 0.29592423161439385, 0.876330778510274, 0.4992998024386687,
                    0.39952552843343986, 0.19896162067804987, 0.09919146330181873, 0.7743867869138954,
                    0.6776512967358279, 0.5934337031624168, 0.2935624509512048, 0.8618585574942957]
