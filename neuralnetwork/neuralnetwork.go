package neuralnetwork

import (
	"fmt"
	"math"

	"github.com/assimoes/NN/helpers"
)

type NeuralNetwork struct {
	InputSize          int
	HiddenSize         int
	OutputSize         int
	LearningRate       float64
	Momentum           float64
	A1, A2, A3         []float64
	W1, W2             [][]float64
	Changes1, Changes2 [][]float64
}

func NewNeuralNetwork(i, h, o int, l, m float64) (nn NeuralNetwork) {

	nn.InputSize = i + 1  // +bias
	nn.HiddenSize = h + 1 //+bias

	nn.OutputSize = o
	nn.LearningRate = l
	nn.Momentum = m

	nn.A1 = helpers.MakeVector(nn.InputSize, 1.0)
	nn.A2 = helpers.MakeVector(nn.HiddenSize, 1.0)
	nn.A3 = helpers.MakeVector(nn.OutputSize, 1.0)

	nn.W1 = helpers.RandomMatrix(nn.InputSize, nn.HiddenSize, -1.0, 1.0)
	nn.W2 = helpers.RandomMatrix(nn.HiddenSize, nn.OutputSize, -1.0, 1.0)

	nn.Changes1 = helpers.MakeMatrix(nn.InputSize, nn.HiddenSize)
	nn.Changes2 = helpers.MakeMatrix(nn.HiddenSize, nn.OutputSize)

	return nn
}

type activation func(float64) float64

func forwardActivation(nrows, ncols int, a []float64, b [][]float64, target []float64, act activation) {

	for i := 0; i < nrows; i++ {
		var sum float64
		for j := 0; j < ncols; j++ {
			sum += a[j] * b[j][i]
		}

		target[i] = act(sum)
	}
}
func updateWeights(nrows, ncols int, deltas []float64, activatedNeuron []float64, weight [][]float64, changes [][]float64, lrate, momentum float64) {

	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			change := deltas[j] * activatedNeuron[i]
			weight[i][j] = weight[i][j] + lrate*change + momentum*changes[i][j]
			changes[i][j] = change
		}
	}

}
func computeErrors(outputSize, hiddenSize int, output, hypothesis, layer2activation []float64, layer2weights [][]float64) ([]float64, []float64) {
	delta2 := helpers.MakeVector(hiddenSize, 0.0)
	delta3 := helpers.MakeVector(outputSize, 0.0)

	for i := 0; i < outputSize; i++ {
		delta3[i] = helpers.DSigmoid(hypothesis[i]) * (output[i] - hypothesis[i])
	}

	for i := 0; i < hiddenSize; i++ {
		var err float64
		for j := 0; j < outputSize; j++ {
			err += delta3[j] * layer2weights[i][j]
		}

		delta2[i] = helpers.DSigmoid(layer2activation[i]) * err
	}

	return delta2, delta3
}
func (nn *NeuralNetwork) Forward(X []float64) []float64 {

	// no caso da função de activação dos neuronios de entrada (a1), o valor é a própria entrada
	for i := 0; i < nn.InputSize-1; i++ {
		nn.A1[i] = X[i]
	}

	// calcula a função de activação da hidden layer (a2)
	forwardActivation(nn.HiddenSize-1, nn.InputSize, nn.A1, nn.W1, nn.A2, helpers.Sigmoid)

	// calcula a função de activação da output layer (a3) ou hipótese.
	forwardActivation(nn.OutputSize, nn.HiddenSize, nn.A2, nn.W2, nn.A3, helpers.Sigmoid)

	return nn.A3
}
func (nn *NeuralNetwork) Backpropagate(output []float64) float64 {

	// calcula erros que serão propagados para corrigir as sinapses
	delta2, delta3 := computeErrors(nn.OutputSize, nn.HiddenSize, output, nn.A3, nn.A2, nn.W2)
	// actualiza w2
	updateWeights(nn.HiddenSize, nn.OutputSize, delta3, nn.A2, nn.W2, nn.Changes2, nn.LearningRate, nn.Momentum)
	//actualiza w1
	updateWeights(nn.InputSize, nn.HiddenSize, delta2, nn.A1, nn.W1, nn.Changes1, nn.LearningRate, nn.Momentum)

	// calcula erro quadrado total desta previsão
	var J float64
	for i := 0; i < len(output); i++ {
		J += 0.5 * math.Pow(output[i]-nn.A3[i], 2)
	}

	return J
}
func (nn *NeuralNetwork) Train(dataset [][][]float64, iterations int) {
	for i := 0; i < iterations; i++ {

		var j float64

		for _, ds := range dataset {

			hypothesis := nn.Forward(ds[0])
			e := nn.Backpropagate(ds[1])
			j += e

			if i%1000 == 0 {
				fmt.Println(i, e, hypothesis, ds[1])
			}
		}

	}
}
func (nn *NeuralNetwork) Predict(X []float64) []float64 {
	return nn.Forward(X)
}
