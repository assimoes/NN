package main

import (
	"fmt"

	nnet "github.com/assimoes/NN/neuralnetwork"
	p "github.com/assimoes/NN/persistence"
	redis "github.com/assimoes/NN/redis"
)

func main() {

	loader := p.NewNetworkLoader()
	nn := loader.Load("nn2.json")

	dataset := []float64{.1, .2}

	fmt.Println(nn.Predict(dataset))

	redis.Run()
}

func fromScratch() {
	nn := nnet.NewNeuralNetwork(2, 4, 1, 0.001, 0.6)

	dataset := [][][]float64{
		{{.1, .2}, {.3}},
		{{.3, .4}, {.7}},
		{{.5, .5}, {1.}},
		{{.2, .2}, {.4}},
	}

	nn.Train(dataset, 4000)

	w := p.NewNetworkWriter()
	w.Save(nn, "nn.json")
}
