package persistence

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	nn "github.com/assimoes/NN/neuralnetwork"
)

type NetworkWriter struct{}

func NewNetworkWriter() NetworkWriter {
	return NetworkWriter{}
}

func (w *NetworkWriter) Save(obj nn.NeuralNetwork, filename string) bool {

	b, err := json.Marshal(obj)

	if err != nil {
		return false
	}

	fmt.Println(string(b))

	e := ioutil.WriteFile(filename, b, 0644)
	check(e)
	return true
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}
