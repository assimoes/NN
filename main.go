package main

import (
	"fmt"

	nnet "github.com/assimoes/NN/neuralnetwork"
)

func main() {

	nn := nnet.NewNeuralNetwork(2, 4, 1, 0.001, 0.06)

	patterns := [][][]float64{
		{{.1, .2}, {.3}},
		{{.3, .4}, {.7}},
		{{.5, .5}, {1.}},
		{{.2, .2}, {.4}},
	}

	for i := 0; i < 4000; i++ {

		var j float64

		for _, p := range patterns {

			hypothesis := nn.Forward(p[0])
			e := nn.Backpropagate(p[1], 1, 0)
			j += e

			if i%100 == 0 {
				fmt.Println(i, e, hypothesis, p[1])
			}
		}

	}

}
