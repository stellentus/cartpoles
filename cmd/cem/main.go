package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"runtime"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/util/lockweight"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

var (
	cpus = flag.Int("cpus", 2, "Number of CPUs")
)

func main() {
	flag.Parse()

	runtime.GOMAXPROCS(*cpus) // Limit the number of CPUs to the provided value (unchanged if the input is <1)

	debug := logger.NewDebug(logger.DebugConfig{}) // TODO create a debug

	numTimesteps := 1000000
	numRuns := 1
	hyperparams := [5]string{"tilings", "tiles", "lambda", "epsilon", "alpha"}
	lower := [len(hyperparams)]float64{0.5, 0.5, 0.0, 0.0, 0.0}
	upper := [len(hyperparams)]float64{7.5, 4.5, 1.0, 1.0, 10}

	//typeOfHyperparams := [5]string{"discrete", "discrete", "continuous", "continuous", "continuous"}
	discreteHyperparamsIndices := [2]int64{0, 1}
	discreteRanges := [][]float64{[]float64{1, 2, 4, 8, 16, 32}, []float64{1, 2, 4}}
	discreteMidRanges := [][]float64{[]float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, []float64{1.5, 2.5, 3.5}}

	numSamples := 50 // 300
	percentElite := 0.5
	numElite := int64(float64(numSamples) * percentElite)
	e := math.Pow(10, -8)
	iterations := 0

	var meanHyperparams [len(hyperparams)]float64
	for i := range meanHyperparams {
		meanHyperparams[i] = (lower[i] + upper[i]) / 2.0
	}

	//covariance := make([][]float64, len(hyperparams))
	//for i := range covariance {
	//	covariance[i] = make([]float64, len(hyperparams))
	//}

	covariance := mat.NewDense(len(hyperparams), len(hyperparams), nil)
	covarianceRows, covarianceColumns := covariance.Dims()

	minLower := lower[0]
	maxUpper := upper[0]

	for _, v := range lower {
		if v < minLower {
			minLower = v
		}
	}
	for _, v := range upper {
		if v > maxUpper {
			maxUpper = v
		}
	}

	for i := 0; i < covarianceRows; i++ {
		for j := 0; j < covarianceColumns; j++ {
			if i == j {
				covariance.Set(i, j, math.Pow(maxUpper-minLower, 2)+e)
			} else {
				covariance.Set(i, j, e)
			}
		}
	}

	fmt.Println("Before")
	matPrint(covariance)
	covariance = nearestPD(covariance)
	fmt.Println("After")
	matPrint(covariance)

	symmetricCovariance := mat.NewSymDense(len(hyperparams), nil)
	for i := 0; i < len(hyperparams); i++ {
		for j := 0; j < len(hyperparams); j++ {
			symmetricCovariance.SetSym(i, j, (covariance.At(i, j)+covariance.At(j, i))/2.0)
		}
	}

	fmt.Println("Symmetric")
	matPrint(symmetricCovariance)

	var choleskySymmetricCovariance mat.Cholesky
	choleskySymmetricCovariance.Factorize(symmetricCovariance)

	samples := make([][]float64, numSamples)
	realvaluedSamples := make([][]float64, numSamples)
	var src rand.Source
	i := 0

	// CODE IS BUGGY RIGHT NOW
	for i < numSamples {
		sample := distmv.NormalRand(nil, meanHyperparams[:], &choleskySymmetricCovariance, src)
		flag := 0
		for j := 0; j < len(hyperparams); j++ {
			if sample[j] < lower[j] || sample[j] > upper[j] {
				flag = 1
				break
			}
		}
		if flag == 0 {
			realvaluedSamples[i] = sample
			var temp []float64
			for j := 0; j < len(hyperparams); j++ {
				if containsInt(discreteHyperparamsIndices[:], int64(j)) {
					for k := 0; k < len(discreteMidRanges); k++ {
						if sample[j] <= discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k] {
							temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k])
							break
						}
					}
					if sample[j] > discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][len(discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])])-1] {
						temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][len(discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])])-1])
					}

				} else {
					temp = append(temp, sample[j])
				}
			}
			samples[i] = temp
			i++
		}

	}

	// to create MVND you need to use gonum NewNormal and covariance = (covariance + covariance.T)/2.0 to make it symmetric
	for i := 0; i < numSamples; i++ {
		fmt.Println(realvaluedSamples[i])
		fmt.Println(samples[i])
		fmt.Println("")
		fmt.Println("")
	}

	fmt.Println("-------------------")
	fmt.Println("-------------------")
	fmt.Println(numTimesteps, numRuns, hyperparams, lower, upper, discreteHyperparamsIndices, discreteMidRanges, discreteRanges, numSamples, numElite, e, iterations, meanHyperparams, covariance)

	//iterations
	//    samples
	//        runs
	for {
		agentSettings := agent.DefaultESarsaSettings()
		// Do CEM stuff to change settings and SEED

		ag := &agent.ESarsa{Debug: debug}
		ag.InitializeWithSettings(agentSettings, lockweight.LockWeight{})

		env := &environment.Cartpole{Debug: debug}
		env.InitializeWithSettings(environment.CartpoleSettings{Seed: int64(0)}) // TODO change seed

		data, err := logger.NewData(debug, logger.DataConfig{
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			BasePath:                "",
			FileSuffix:              "",
		})
		panicIfError(err, "Couldn't create logger.Data")

		expConf := config.Experiment{
			MaxEpisodes:             0,
			MaxSteps:                10,
			DebugInterval:           0,
			DataPath:                "",
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			MaxCPUs:                 *cpus,
		}
		exp, err := experiment.New(ag, env, expConf, debug, data)
		panicIfError(err, "Couldn't create experiment")

		listOfRewards, _ := exp.Run()
		fmt.Println(listOfRewards)

		//fmt.Println(data.NumberOfEpisodes())
		break
	}
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}

func nearestPD(A *mat.Dense) *mat.Dense {
	ARows, AColumns := A.Dims()
	//fmt.Println("A")
	//matPrint(A)
	//fmt.Println("B")

	B := mat.NewDense(ARows, AColumns, nil)

	transposedA := transpose(A)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			value := (A.At(i, j) + transposedA.At(i, j)) / 2.0
			B.Set(i, j, value)
		}
	}
	//matPrint(B)

	u, s, v, err := svd(B)
	//fmt.Println("U")
	//matPrint(&u)
	//fmt.Println("s")
	//fmt.Println(s)
	//fmt.Println("V")
	//matPrint(&v)

	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("")
		//fmt.Println(u)
		//fmt.Println(s)
		//fmt.Println(v)

	}

	uRows, uColumns := u.Dims()
	vRows, vColumns := v.Dims()
	uDense := mat.NewDense(uRows, uColumns, nil)
	vDense := mat.NewDense(vRows, vColumns, nil)

	for i := 0; i < uRows; i++ {
		for j := 0; j < uColumns; j++ {
			uDense.Set(i, j, u.At(i, j))
		}
	}

	for i := 0; i < vRows; i++ {
		for j := 0; j < vColumns; j++ {
			vDense.Set(i, j, v.At(i, j))
		}
	}

	diagonalSMatrix := mat.NewDense(ARows, AColumns, nil)
	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			if i == j {
				diagonalSMatrix.Set(i, j, s[i])
			} else {
				diagonalSMatrix.Set(i, j, 0.0)
			}

		}
	}

	var temp mat.Dense
	var original mat.Dense
	temp.Mul(uDense, diagonalSMatrix)
	original.Mul(&temp, transpose(vDense))
	originalRows, originalColumns := original.Dims()
	originalDense := mat.NewDense(originalRows, originalColumns, nil)

	for i := 0; i < originalRows; i++ {
		for j := 0; j < originalColumns; j++ {
			originalDense.Set(i, j, original.At(i, j))
		}
	}
	//fmt.Println("Original")
	//matPrint(originalDense)

	var temp0 mat.Dense
	var H mat.Dense
	temp0.Mul(diagonalSMatrix, vDense)
	H.Mul(transpose(vDense), &temp0)
	hRows, hColumns := H.Dims()
	hDense := mat.NewDense(hRows, hColumns, nil)

	for i := 0; i < hRows; i++ {
		for j := 0; j < hColumns; j++ {
			hDense.Set(i, j, H.At(i, j))
		}
	}

	//fmt.Println("H")
	//matPrint(hDense)

	A2 := mat.NewDense(ARows, AColumns, nil)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A2.Set(i, j, (B.At(i, j)+hDense.At(i, j))/2.0) // it is H[i][j]
		}
	}

	//fmt.Println("A2")
	//matPrint(A2)
	A3 := mat.NewDense(ARows, AColumns, nil)

	transposedA2 := transpose(A2)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A3.Set(i, j, (A2.At(i, j)+transposedA2.At(i, j))/2.0)
		}
	}

	//fmt.Println("A3")
	//matPrint(A3)
	//fmt.Println(A3)

	if isPD(A3) {
		return A3
	}

	normA := mat.Norm(A, 2)
	nextNearestDistance := math.Nextafter(normA, normA+1) - normA
	previousNearestDistance := normA - math.Nextafter(normA, normA-1)

	spacing := math.Min(nextNearestDistance, previousNearestDistance)

	I := mat.NewDense(ARows, ARows, nil)
	for i := 0; i < ARows; i++ {
		for j := 0; j < ARows; j++ {
			if i == j {
				I.Set(i, j, 1)
			} else {
				I.Set(i, j, 0)
			}
		}
	}

	k := 1.0

	for !isPD(A3) {

		var eig mat.Eigen
		ok := eig.Factorize(A3, mat.EigenNone)
		if !ok {
			fmt.Println("Eigen decomposition failed")
		}
		eigenvalues := eig.Values(nil)

		realeigenvalues := make([]float64, len(eigenvalues))
		for i := range eigenvalues {
			realeigenvalues[i] = real(eigenvalues[i])
		}

		minrealeigenvalues := realeigenvalues[0]
		for _, value := range realeigenvalues {
			if minrealeigenvalues > value {
				minrealeigenvalues = value
			}
		}
		for i := 0; i < ARows; i++ {
			for j := 0; j < AColumns; j++ {
				A3.Set(i, j, I.At(i, j)*((-minrealeigenvalues*math.Pow(k, 2))+spacing))
			}
		}
		k++
	}

	return A3
}

func isPD(matrix mat.Matrix) bool {
	boolean := true

	var eig mat.Eigen
	ok := eig.Factorize(matrix, mat.EigenNone)
	if !ok {
		fmt.Println("Eigen decomposition failed")
	}
	eigenvalues := eig.Values(nil)

	realeigenvalues := make([]float64, len(eigenvalues))
	for i := range eigenvalues {
		realeigenvalues[i] = real(eigenvalues[i])
	}

	for i := range realeigenvalues {
		if realeigenvalues[i] <= 0 {
			boolean = false
			break
		}
	}

	return boolean
}

func transpose(matrix *mat.Dense) *mat.Dense {
	rows, columns := matrix.Dims()
	transposeMatrix := mat.NewDense(rows, columns, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			transposeMatrix.Set(i, j, matrix.At(j, i))
		}
	}
	return transposeMatrix
}

func svd(Matrix mat.Matrix) (mat.Dense, []float64, mat.Dense, error) {
	var svd mat.SVD
	if ok := svd.Factorize(Matrix, mat.SVDFull); !ok {
		var nilMat mat.Dense
		return nilMat, nil, nilMat, errors.New("SVD factorization failed")
	}
	var v, u mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	s := svd.Values(nil)
	return u, s, v, nil
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func containsInt(s []int64, e int64) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

func indexOfInt(element int64, data []int64) int {
	for k, v := range data {
		if element == v {
			return k
		}
	}
	return -1 //not found.
}
