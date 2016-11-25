package main

import nnet "github.com/assimoes/NN/neuralnetwork"

func main() {

	nn := nnet.NewNeuralNetwork(2, 4, 1, 0.001, 0.6)

	dataset := [][][]float64{
		{{.1, .2}, {.3}},
		{{.3, .4}, {.7}},
		{{.5, .5}, {1.}},
		{{.2, .2}, {.4}},
	}

	nn.Train(dataset, 4000)
}
