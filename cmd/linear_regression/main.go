package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	linear_regression "github.com/careyi3/go-ml/cmd/linear_regression/internal"
	"github.com/careyi3/go-ml/internal/models"

	"github.com/jedib0t/go-pretty/v6/table"
)

var alphaFlag float64
var lambdaFlag float64
var itersFlag int
var inputFlag string
var outputFlag string

func init() {
	flag.Float64Var(&alphaFlag, "a", 0.9, "Learning Rate (alpha)")
	flag.Float64Var(&lambdaFlag, "l", 0.0, "Regularization Constant (lambda)")
	flag.IntVar(&itersFlag, "i", 50000, "# of iterations of gradient descent for learning")
	flag.StringVar(&inputFlag, "in", "./input.csv", "Input CSV file")
	flag.StringVar(&outputFlag, "out", "./", "Output directory")
}

func main() {
	flag.Parse()
	exitCode := 0
	fmt.Println("Training Started.")
	start := time.Now()
	results, err := linear_regression.Run(inputFlag, outputFlag, alphaFlag, lambdaFlag, itersFlag)
	if err != nil {
		elapsed := time.Since(start)
		exitCode = 1
		fmt.Printf("%v\n", err)
		fmt.Printf("Duration: %s", elapsed)
	}
	prettyPrintResults(*results)
	fmt.Println("Training Done.")
	elapsed := time.Since(start)
	fmt.Printf("Duration: %s\n", elapsed)
	os.Exit(exitCode)
}

func prettyPrintResults(results models.TrainingResults) {
	t := table.NewWriter()
	t.SetOutputMirror(os.Stdout)
	t.AppendHeader(table.Row{"Training Error ('%')", "CV Error ('%')", "Test Error ('%')"})
	t.AppendRows([]table.Row{
		{results.TrainingError, results.CVError, results.TestError},
	})
	t.Render()
}
