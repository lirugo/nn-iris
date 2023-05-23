from NeuralNetwork import NeuralNetwork


def main():
    nn = NeuralNetwork(12) # Number of neuron inside hidden layer
    nn.train(13_000) # Number of iteration

    print('-------------------------')

    setosa = nn.feed_forward([5.1, 3.5, 1.4, 0.2])
    versicolor = nn.feed_forward([7.0, 3.2, 4.7, 1.4])
    viginica = nn.feed_forward([6.3, 3.3, 6.0, 2.5])

    print('#0 Iris-setosa:', (setosa.index(max(setosa))))
    print('#1 Iris-versicolor:', (versicolor.index(max(versicolor))))
    print('#2 Iris-virginica:', (viginica.index(max(viginica))))
    print('-------------------------')


if __name__ == "__main__":
    main()
