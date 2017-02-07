package main

import (
	"fmt"
	"log"
	"net/http"

	api "github.com/assimoes/NN/api"
	nnet "github.com/assimoes/NN/neuralnetwork"
	p "github.com/assimoes/NN/persistence"
)

func main() {

	loader := p.NewNetworkLoader()
	nn := loader.Load("nn2.json")

	dataset := []float64{.1, .2}

	fmt.Println(nn.Predict(dataset))

	router := api.NewRouter()
	log.Fatal(http.ListenAndServe(":8080", router))

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
