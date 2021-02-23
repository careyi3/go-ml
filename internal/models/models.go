package models

import "gonum.org/v1/gonum/mat"

type DataSet struct {
	X mat.Dense
	Y mat.Dense
	R mat.Dense
	K mat.Dense
}

type ParsedInputs struct {
	TrainingSet DataSet
	CVSet       DataSet
	TestSet     DataSet
	Theta       mat.Dense
}

type TrainingResults struct {
	Thetas        []float64
	Alpha         float64
	Lambda        float64
	NumIterations int
	TrainingError float64
	CVError       float64
	TestError     float64
}
