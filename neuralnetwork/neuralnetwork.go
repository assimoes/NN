package neuralnetwork

import (
	"math"

	"github.com/assimoes/NN/helpers"
)

type NeuralNetwork struct {
	inputSize          int
	hiddenSize         int
	outputSize         int
	learningRate       float64
	momentum           float64
	a1, a2, a3         []float64
	w1, w2             [][]float64
	changes1, changes2 [][]float64
}

func NewNeuralNetwork(i, h, o int, l, m float64) (nn NeuralNetwork) {

	nn.inputSize = i + 1  // +bias
	nn.hiddenSize = h + 1 //+bias

	nn.outputSize = o
	nn.learningRate = l
	nn.momentum = m

	nn.a1 = helpers.MakeVector(nn.inputSize, 1.0)
	nn.a2 = helpers.MakeVector(nn.hiddenSize, 1.0)
	nn.a3 = helpers.MakeVector(nn.outputSize, 1.0)

	nn.w1 = helpers.RandomMatrix(nn.inputSize, nn.hiddenSize, -1.0, 1.0)
	nn.w2 = helpers.RandomMatrix(nn.hiddenSize, nn.outputSize, -1.0, 1.0)

	nn.changes1 = helpers.MakeMatrix(nn.inputSize, nn.hiddenSize)
	nn.changes2 = helpers.MakeMatrix(nn.hiddenSize, nn.outputSize)

	return nn
}

func (nn *NeuralNetwork) Forward(X []float64) []float64 {

	// no caso da função de activação dos neuronios de entrada (a1), o valor é a própria entrada
	for i := 0; i < nn.inputSize-1; i++ {
		nn.a1[i] = X[i]
	}

	// calcula a função de activação da hidden layer (a2)
	for i := 0; i < nn.hiddenSize-1; i++ {
		var sum float64
		for j := 0; j < nn.inputSize; j++ {
			sum += nn.a1[j] * nn.w1[j][i]
		}

		nn.a2[i] = helpers.Sigmoid(sum)
	}

	// calcula a função de activação da output layer (a3) ou hipótese.

	for i := 0; i < nn.outputSize; i++ {
		var sum float64
		for j := 0; j < nn.hiddenSize; j++ {
			sum += nn.a2[j] * nn.w2[j][i]
		}

		nn.a3[i] = helpers.Sigmoid(sum)
	}

	return nn.a3
}

func (nn *NeuralNetwork) Backpropagate(output []float64, learningRate, momentum float64) float64 {

	a3Deltas := helpers.MakeVector(nn.outputSize, 0.0)
	a2Deltas := helpers.MakeVector(nn.hiddenSize, 0.0)

	// calcula deltas entre output e hipotese e aplica a derivada parcial da função sigmoid original
	for i := 0; i < nn.outputSize; i++ {
		a3Deltas[i] = helpers.DSigmoid(nn.a3[i]) * (output[i] - nn.a3[i])
	}

	// calcula erro
	for i := 0; i < nn.hiddenSize; i++ {
		var err float64
		for j := 0; j < nn.outputSize; j++ {
			err += a3Deltas[j] * nn.w2[i][j]
		}

		a2Deltas[i] = helpers.DSigmoid(nn.a2[i]) * err
	}

	// actualiza w2
	for i := 0; i < nn.hiddenSize; i++ {
		for j := 0; j < nn.outputSize; j++ {
			change := a3Deltas[j] * nn.a2[i]
			nn.w2[i][j] = nn.w2[i][j] + learningRate*change + momentum*nn.changes2[i][j]
			nn.changes2[i][j] = change
		}
	}

	//actualiza w1
	for i := 0; i < nn.inputSize; i++ {
		for j := 0; j < nn.hiddenSize; j++ {
			change := a2Deltas[j] * nn.a1[i]
			nn.w1[i][j] = nn.w1[i][j] + learningRate*change + momentum*nn.changes1[i][j]
			nn.changes1[i][j] = change
		}
	}

	// calcula erro quadrado total desta previsão
	var J float64
	for i := 0; i < len(output); i++ {
		J += 0.5 * math.Pow(output[i]-nn.a3[i], 2)
	}

	return J
}
