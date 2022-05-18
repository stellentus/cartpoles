package environment

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"


	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/stellentus/cartpoles/lib/util/convformat"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type ReadSettings struct {
	DataLog      	   string  `json:"datalog"`
	Seed               int64   `json:"seed"`
	TotalLogs          uint    `json:"total-logs"`
}

type ReadDataEnv struct {
	logger.Debug
	state rlglue.State
	rng   *rand.Rand
	ReadSettings
	LoadedLog string

	stateDim int
	NumberOfActions int
	offlineDataObs  [][]float64
	dataCount int
	dataLines int
}

func init() {
	Add("readdata", NewReadDataEnv)
}

func NewReadDataEnv(logger logger.Debug) (rlglue.Environment, error) {
	return &ReadDataEnv{Debug: logger}, nil
}

func (env *ReadDataEnv) SettingFromLog(paramLog string) {
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
	for scanner.Scan() {
		line = scanner.Text()
		spl = strings.Split(line, "=")
		if spl[0] == "stateDimension" { //stateDimension
			stateDim, _ = strconv.Atoi(spl[1])
		} else if spl[0] == "numberOfActions" {
			numAction, _ = strconv.Atoi(spl[1])
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	env.stateDim = stateDim
	env.NumberOfActions = numAction
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *ReadDataEnv) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.ReadSettings)
	if err != nil {
		err = errors.New("environment.knnModel settings error: " + err.Error())
		env.Message("err", err)
		return err
	}

	env.ReadSettings.Seed += int64(run / env.ReadSettings.TotalLogs)
	// For CEM, use env.ReadSettings.Seed += int64(run)

	env.rng = rand.New(rand.NewSource(env.ReadSettings.Seed)) // Create a new rand source for reproducibility

	env.Message("environment.knnModel settings", fmt.Sprintf("%+v", env.ReadSettings))

	folder := env.ReadSettings.DataLog
	env.LoadedLog = strconv.Itoa(int(run%env.ReadSettings.TotalLogs))
	traceLog := folder + "/traces-" + env.LoadedLog + ".csv"
	//fmt.Println(traceLog)
	//fmt.Println(trueStartLog)

	env.Message("KNN data log", traceLog, "\n")
	paramLog := folder + "/log_json.txt"
	//env.Message("KNN json log", paramLog, "\n")
	env.SettingFromLog(paramLog)
	env.state = make(rlglue.State, env.stateDim)

	//env.offlineData = allTrans
	env.offlineDataObs = env.LoadData(traceLog)
	env.dataLines = len(env.offlineDataObs)
	env.dataCount = 0
	return nil
}

func (env *ReadDataEnv) LoadData(filename string) ([][]float64) {
	// Get offline data
	csvFile, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	allTransStr, err := csv.NewReader(csvFile).ReadAll()
	csvFile.Close()
	if err != nil {
		log.Fatal(err)
	}

	allTransObs := make([][]float64, len(allTransStr)-1) // transitions, using environment state
	rewards := make([]float64, len(allTransStr)-1)
	allStates := make([][]float64, len(allTransStr)-1)     // current states
	allNextStates := make([][]float64, len(allTransStr)-1) // next states
	allTermin := make([]float64, len(allTransStr)-1)
	var allReturns []float64
	rTemp := 0.0
	for i := 1; i < len(allTransStr); i++ { // remove first str (title of column)
		trans := allTransStr[i]
		row := make([]float64, env.stateDim*2+3)
		for j, num := range trans {
			if j == 0 { // next state
				num = num[1 : len(num)-1] // remove square brackets
				copy(row[env.stateDim+1:env.stateDim*2+1], convformat.ListStr2Float(num, " "))

				allNextStates[i-1] = make([]float64, env.stateDim)
				copy(allNextStates[i-1], row[env.stateDim+1:env.stateDim*2+1])

			} else if j == 1 { // current state
				num = num[1 : len(num)-1]
				copy(row[:env.stateDim], convformat.ListStr2Float(num, " "))

				allStates[i-1] = make([]float64, env.stateDim)
				copy(allStates[i-1], row[:env.stateDim])

			} else if j == 2 { // action
				row[env.stateDim], _ = strconv.ParseFloat(num, 64)

			} else if j == 3 { //reward
				row[env.stateDim*2+1], _ = strconv.ParseFloat(num, 64)
				rewards[i-1] = row[env.stateDim*2+1]
				rTemp += rewards[i-1]

			} else if j == 4 { //termination
				row[env.stateDim*2+2], _ = strconv.ParseFloat(num, 64)
				allTermin[i-1], _ = strconv.ParseFloat(num, 64)
				if allTermin[i-1] == 1 {
					allReturns = append(allReturns, rTemp)
					rTemp = 0
				}
			}
		}
		allTransObs[i-1] = row
	}

	return allTransObs
}

// Start returns an initial observation.
func (env *ReadDataEnv) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	env.state = env.offlineDataObs[env.dataCount][:env.stateDim]
	state_copy := make([]float64, env.stateDim)
	copy(state_copy, env.state)
	return state_copy, ""
}

func (env *ReadDataEnv) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	//actInt, _ := tpo.GetInt(act)

	rec := env.offlineDataObs[env.dataCount]
	//s := rec[:env.stateDim]
	//a := rec[env.stateDim]
	sp := rec[env.stateDim+1: env.stateDim*2+1]
	reward := rec[env.stateDim*2+1]
	done := rec[env.stateDim*2+2] == 1
	//fmt.Println("ENV", rec, env.dataCount)

	env.dataCount = (env.dataCount + 1) % env.dataLines

	env.state = sp
	state_copy := make([]float64, env.stateDim)
	copy(state_copy, sp)

	return state_copy, reward, done, ""
}

// GetAttributes returns attributes for this environment.
func (env *ReadDataEnv) GetAttributes() rlglue.Attributes {
	return rlglue.Attributes(`{"numberOfActions":4,"state-contains-replay":true}`)
	// TODO should be saved as attributes from a known struct
}

func (env *ReadDataEnv) GetInfo(info string, value float64) interface{} {
	return nil
}