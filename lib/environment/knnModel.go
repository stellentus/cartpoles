package environment

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/stellentus/cartpoles/lib/util/convformat"
	"github.com/stellentus/cartpoles/lib/util/random"
	transModel "github.com/stellentus/cartpoles/lib/util/transkdtree"
	tpo "github.com/stellentus/cartpoles/lib/util/type-opr"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type knnSettings struct {
	DataLog      string  `json:"datalog"`
	Seed         int64   `json:"seed"`
	Neighbor_num int     `json:"neighbor-num"`
	EnsembleSeed int     `json:"ensemble-seed"`
	DropPerc     float64 `json:"drop-percent"`
	//Timeout      int 	 `json:"timeout"`
}

type KnnModelEnv struct {
	logger.Debug
	knnSettings
	state           rlglue.State
	rng             *rand.Rand
	offlineData     [][]float64
	offlineStarts   []int
	offlineModel    transModel.TransTrees
	stateDim        int
	NumberOfActions int
	stateRange      []float64
	neighbor_prob   []float64
	rewardBound 	[]float64
	//Count			int

	DebugArr   [][]float64
}

func init() {
	Add("knnModel", NewKnnModelEnv)
}

func NewKnnModelEnv(logger logger.Debug) (rlglue.Environment, error) {
	return &KnnModelEnv{Debug: logger}, nil
}

func (env *KnnModelEnv) SettingFromLog(paramLog string) {
	// Get state dimension
	txt, err := os.Open(paramLog)
	if err != nil {
		panic(err)
	}
	defer txt.Close()
	scanner := bufio.NewScanner(txt)
	var line string
	var spl []string
	var stateDim int
	var numAction int
	var stateRange []float64
	for scanner.Scan() {
		line = scanner.Text()
		spl = strings.Split(line, "=")
		if spl[0] == "stateDimension" { //stateDimension
			stateDim, _ = strconv.Atoi(spl[1])
		} else if spl[0] == "numberOfActions" {
			numAction, _ = strconv.Atoi(spl[1])
		} else if spl[0] == "stateRange" {
			stateRange = convformat.ListStr2Float(spl[1], ",")
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	env.stateDim = stateDim
	env.NumberOfActions = numAction
	env.stateRange = stateRange
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *KnnModelEnv) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.knnSettings)
	if err != nil {
		err = errors.New("environment.knnModel settings error: " + err.Error())
		env.Message("err", err)
		return err
	}
	env.Seed += int64(run)
	env.rng = rand.New(rand.NewSource(env.Seed)) // Create a new rand source for reproducibility
	env.state = make(rlglue.State, 4)
	//env.Count = 0

	env.Message("environment.knnModel settings", fmt.Sprintf("%+v", env.knnSettings))

	folder := env.knnSettings.DataLog
	traceLog := folder + "/traces-" + strconv.Itoa(int(run)) + ".csv"
	paramLog := folder + "/log_json.txt"
	env.SettingFromLog(paramLog)

	// Get offline data
	csvFile, err := os.Open(traceLog)
	if err != nil {
		log.Fatal(err)
	}
	allTransStr, err := csv.NewReader(csvFile).ReadAll()
	csvFile.Close()
	if err != nil {
		log.Fatal(err)
	}
	allTransTemp := make([][]float64, len(allTransStr)-1)
	rewards := make([]float64, len(allTransStr)-1)
	for i := 1; i < len(allTransStr); i++ { // remove first str (title of column)
		trans := allTransStr[i]
		row := make([]float64, env.stateDim*2+3)
		for j, num := range trans {
			if j == 0 { // next state
				num = num[1 : len(num)-1] // remove square brackets
				copy(row[env.stateDim+1:env.stateDim*2+1], convformat.ListStr2Float(num, " "))
			} else if j == 1 { // current state
				num = num[1 : len(num)-1]
				copy(row[:env.stateDim], convformat.ListStr2Float(num, " "))
			} else if j == 2 { // action
				row[env.stateDim], _ = strconv.ParseFloat(num, 64)
			} else if j == 3 { //reward
				row[env.stateDim*2+1], _ = strconv.ParseFloat(num, 64)
				rewards[i-1] = row[env.stateDim*2+1]
				//if row[env.stateDim*2+1] == -1 { // termination
				//	row[env.stateDim*2+2] = 1
				//} else {
				//	row[env.stateDim*2+2] = 0
				//}
			} else if j == 4 { //termination
				row[env.stateDim*2+2], _ = strconv.ParseFloat(num, 64)
			}
		}
		allTransTemp[i-1] = row
	}
	env.rewardBound = make([]float64, 2)
	env.rewardBound[0], _ = ao.ArrayMin(rewards)
	env.rewardBound[1], _ = ao.ArrayMax(rewards)

	var allTrans [][]float64
	if env.EnsembleSeed != 0 {
		tempRnd := rand.New(rand.NewSource(int64(env.EnsembleSeed)))
		filteredLen := int(float64(len(allTransTemp)) * (1 - env.DropPerc))
		filteredIdx := tempRnd.Perm(len(allTransTemp))[:filteredLen]
		allTrans = make([][]float64, filteredLen)
		for i := 0; i < filteredLen; i++ {
			allTrans[i] = allTransTemp[filteredIdx[i]]
		}
	} else {
		allTrans = allTransTemp
	}

	env.offlineData = allTrans
	env.offlineStarts = env.SearchOfflineStart(allTrans)
	env.offlineModel = transModel.New(env.NumberOfActions, env.stateDim)
	env.offlineModel.BuildTree(allTrans)

	//pdf := make([]float64, env.Neighbor_num)
	//temp := 0.0
	//for i := 0; i < env.Neighbor_num; i++ {
	//	pdf[i] = math.Pow(0.5, float64(i+1))
	//	temp += pdf[i]
	//}
	//for i := 0; i < env.Neighbor_num; i++ {
	//	pdf[i] = pdf[i] / temp
	//}
	//env.neighbor_prob = make([]float64, env.Neighbor_num)
	//temp1 := 0.0
	//for i := 0; i < env.Neighbor_num; i++ {
	//	env.neighbor_prob[i] = temp1 + pdf[i]
	//	temp1 = env.neighbor_prob[i]
	//}
	return nil
}

func (env *KnnModelEnv) SearchOfflineStart(allTrans [][]float64) []int {
	starts := []int{0}
	for i := 0; i < len(allTrans)-1; i++ { // not include the end of run
		if allTrans[i][len(allTrans[i])-1] == 1 {
			starts = append(starts, i+1)
		}
	}
	return starts
}

func (env *KnnModelEnv) randomizeInitState() rlglue.State {
	randIdx := env.rng.Intn(len(env.offlineStarts))
	state := env.offlineData[env.offlineStarts[randIdx]][:env.stateDim]
	return state
}

/* For debugging only */
func (env *KnnModelEnv) cheatingChoice() rlglue.State {
	randIdx := env.rng.Intn(len(env.offlineStarts))
	state := env.offlineData[env.offlineStarts[randIdx]-1]
	fmt.Println("terminal ", state)
	state = state[:env.stateDim]
	return state
}

func (env *KnnModelEnv) randomizeState() rlglue.State {
	randIdx := env.rng.Intn(len(env.offlineData))
	state := env.offlineData[randIdx][:env.stateDim]
	return state
}

func (env *KnnModelEnv) furthestState() rlglue.State {
	totalSize := 0
	for act:=0; act<env.NumberOfActions; act++ {
		totalSize += env.offlineModel.TreeSize(act)
	}
	totalStates := make([][]float64, totalSize)
	totalDistance := make([]float64, totalSize)
	idx := 0
	size := 0
	for act:=0; act<env.NumberOfActions; act++ {
		size = env.offlineModel.TreeSize(act)
		states, _, _, _, distances := env.offlineModel.SearchTree(env.state, act, size)
		copy(totalStates[idx: idx+size], states)
		copy(totalDistance[idx: idx+size], distances)
		idx += size
	}
	//fmt.Printf("\n%.2f \n", env.state)

	sum := 0.0
	for i := 0; i < len(totalDistance); i++ {
		sum += totalDistance[i] + math.Pow(10, -6)
	}
	pdf := make([]float64, len(totalDistance))
	for i := 0; i < len(totalDistance); i++ {
		pdf[i] = (totalDistance[i] + math.Pow(10, -6)) / sum
	}
	normalizedPdf := make([]float64, len(totalDistance))
	normalizedSum := 0.0
	for i := 0; i < len(totalDistance); i++ {
		normalizedSum += pdf[i]
	}
	for i := 0; i < len(totalDistance); i++ {
		normalizedPdf[i] = (pdf[i] / normalizedSum)
	}

	prob := make([]float64, len(totalDistance))
	temp1 := 0.0
	for i := 0; i < len(totalDistance); i++ {
		prob[i] = temp1 + normalizedPdf[i]
		temp1 = prob[i]
	}
	chosen := random.FreqSample(prob)
	state := totalStates[chosen]
	//fmt.Printf("%.2f \n", totalDistance[chosen])
	return state
}

// Start returns an initial observation.
func (env *KnnModelEnv) Start() rlglue.State {
	//env.Count = 0
	//env.state = env.randomizeInitState()
	env.state = env.randomizeState()
	//env.state = env.furthestState()
	//env.state = env.cheatingChoice()
	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)
	//fmt.Println("Start", env.state)
	return state_copy
}

func (env *KnnModelEnv) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	//fmt.Println("---------------------")
	actInt, _ := tpo.GetInt(act)

	//env.Count += 1

	if env.offlineModel.TreeSize(actInt)==0 {
		log.Print("Warning: There is no data for action %d, terminating the episode \n", act)
		return env.Start(), env.rewardBound[0], true
	}

	_, nextStates, rewards, terminals, distances := env.offlineModel.SearchTree(env.state, actInt, env.Neighbor_num)

	//fmt.Print("===, ", actInt)
	//fmt.Println(env.offlineModel.SearchTree(env.state, 1, env.Neighbor_num))
	//fmt.Println(env.offlineModel.SearchTree(env.state, 0, env.Neighbor_num))

	sum := 0.0
	for i := 0; i < len(distances); i++ {
		sum += distances[i] + math.Pow(10, -6)
	}
	pdf := make([]float64, len(distances))
	for i := 0; i < len(distances); i++ {
		pdf[i] = 1.0 - ((distances[i] + math.Pow(10, -6)) / sum)
	}
	normalizedPdf := make([]float64, len(distances))
	normalizedSum := 0.0
	for i := 0; i < len(distances); i++ {
		normalizedSum += pdf[i]
	}
	for i := 0; i < len(distances); i++ {
		normalizedPdf[i] = (pdf[i] / normalizedSum)
	}

	env.neighbor_prob = make([]float64, len(distances))
	temp1 := 0.0
	for i := 0; i < len(distances); i++ {
		env.neighbor_prob[i] = temp1 + normalizedPdf[i]
		temp1 = env.neighbor_prob[i]
	}

	chosen := random.FreqSample(env.neighbor_prob)
	env.state = nextStates[chosen]
	reward := rewards[chosen]
	//idx := ao.Search2D(env.state, env.DebugArr)
	//if idx > -1 {
	//	fmt.Printf("%d, %d, %.2f, %d, \n", act, idx, env.DebugArr[idx], terminals[chosen])
	//} else {
	//	fmt.Println(act, idx, terminals[chosen])
	//}
	//env.DebugArr = append(env.DebugArr, env.state)


	var done bool
	if terminals[chosen] == 0 {
		done = false
	} else {
		done = true
	}
	//if env.Timeout!=0 && env.Count>=env.Timeout {
	//	done = true
	//}
	//if done {
	//	env.state = env.Start()
	//}

	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)
	return state_copy, reward, done
}

//GetAttributes returns attributes for this environment.
func (env *KnnModelEnv) GetAttributes() rlglue.Attributes {
	// Add elements to attributes.
	attributes := struct {
		NumAction  int       `json:"numberOfActions"`
		StateDim   int       `json:"stateDimension"`
		StateRange []float64 `json:"stateRange"`
	}{
		env.NumberOfActions,
		env.stateDim,
		env.stateRange,
	}
	attr, err := json.Marshal(&attributes)
	if err != nil {
		env.Message("err", "environment.knnModel could not Marshal its JSON attributes: "+err.Error())
	}
	return attr
}

func (env *KnnModelEnv) Distance(state1 rlglue.State, state2 rlglue.State) float64 {
	var distance float64
	var squareddistance float64
	squareddistance = 0.0
	for i := 0; i < len(state1); i++ {
		squareddistance += math.Pow(state1[i]-state2[i], 2)
	}
	distance = math.Pow(squareddistance, 0.5)
	return distance
}