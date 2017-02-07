package persistence

import (
	"encoding/json"

	"os"

	nn "github.com/assimoes/NN/neuralnetwork"
)

type NetworkLoader struct{}

func NewNetworkLoader() NetworkLoader {
	return NetworkLoader{}
}

func (l *NetworkLoader) Load(filename string) nn.NeuralNetwork {

	data, err := os.Open(filename)
	check(err)

	jsonParser := json.NewDecoder(data)
	n := nn.NeuralNetwork{}
	jsonParser.Decode(&n)

	return n
}
