import random

from Neuron import Neuron


class NeuronLayer:
    def __init__(self, num_neurons, bias):
        self.bias = bias if bias else random.random()
        self.neurons = []

        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

        self.inspect()

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs
