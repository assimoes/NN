package helpers

import (
	"math"
	"math/rand"
)

func MakeVector(size int, fillWith float64) []float64 {
	vector := make([]float64, size)
	for i := 0; i < size; i++ {
		vector[i] = fillWith
	}

	return vector
}

func MakeMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]float64, cols)
	}
	return m
}

func RandomMatrix(rows, cols int, l, u float64) [][]float64 {
	m := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		m[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			m[i][j] = rand.Float64()*(u-l) + l
		}
	}

	return m
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func DSigmoid(y float64) float64 {
	return y * (1 - y)
}
