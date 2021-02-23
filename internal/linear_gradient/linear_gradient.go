package linear_gradient

import (
	"gonum.org/v1/gonum/mat"
)

func Perform(X mat.Dense, Y mat.Dense, Theta mat.Dense, a float64, lambda float64, num_iter int) *mat.Dense {
	m := float64(X.RawMatrix().Rows)
	f := createFilter(Theta)

	var regf mat.Dense
	regf.Scale(a*(lambda/m), f)

	for i := 0; i < num_iter; i++ {
		var pred mat.Dense
		pred.Mul(&X, &Theta)
		var diff mat.Dense
		diff.Sub(&pred, &Y)
		var cost mat.Dense
		cost.Mul(X.T(), &diff)
		var scaled_cost mat.Dense
		scaled_cost.Scale(a/m, &cost)
		var reg_theta mat.Dense
		reg_theta.MulElem(&Theta, &regf)
		scaled_cost.Add(&scaled_cost, &reg_theta)
		Theta.Sub(&Theta, &scaled_cost)
	}

	return &Theta
}

func createFilter(Theta mat.Dense) *mat.Dense {
	f_data := make([]float64, Theta.RawMatrix().Rows)
	for i := 0; i < Theta.RawMatrix().Rows; i++ {
		if i == 0 {
			f_data[i] = 0
		} else {
			f_data[i] = 1
		}
	}

	return mat.NewDense(Theta.RawMatrix().Rows, 1, f_data)
}
