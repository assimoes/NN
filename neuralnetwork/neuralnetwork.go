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
	forwardActivation(nn.hiddenSize-1, nn.inputSize, nn.a1, nn.w1, nn.a2, helpers.Sigmoid)

	// calcula a função de activação da output layer (a3) ou hipótese.
	forwardActivation(nn.outputSize, nn.hiddenSize, nn.a2, nn.w2, nn.a3, helpers.Sigmoid)

	return nn.a3
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

func (nn *NeuralNetwork) Backpropagate(output []float64, learningRate, momentum float64) float64 {

	// calcula erros que serão propagados para corrigir as sinapses
	delta2, delta3 := computeErrors(nn.outputSize, nn.hiddenSize, output, nn.a3, nn.a2, nn.w2)
	// actualiza w2
	updateWeights(nn.hiddenSize, nn.outputSize, delta3, nn.a2, nn.w2, nn.changes2, learningRate, momentum)
	//actualiza w1
	updateWeights(nn.inputSize, nn.hiddenSize, delta2, nn.a1, nn.w1, nn.changes1, learningRate, momentum)

	// calcula erro quadrado total desta previsão
	var J float64
	for i := 0; i < len(output); i++ {
		J += 0.5 * math.Pow(output[i]-nn.a3[i], 2)
	}

	return J
}
