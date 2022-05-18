package agent

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/buffer"
	"github.com/stellentus/cartpoles/lib/util/convformat"
	"github.com/stellentus/cartpoles/lib/util/lockweight"
	"github.com/stellentus/cartpoles/lib/util/network"
	"github.com/stellentus/cartpoles/lib/util/normalizer"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
)

type cqlSettings struct {
	Seed        int64
	EnableDebug bool `json:"enable-debug"`

	NumberOfActions     int     `json:"numberOfActions"`
	StateContainsReplay bool    `json:"state-contains-replay"`
	Gamma               float64 `json:"gamma"`
	Epsilon             float64 `json:"epsilon"`
	MinEpsilon          float64 `json:"min-epsilon"`
	DecreasingEpsilon   string  `json:"decreasing-epsilon"`

	NumDataset int64   `json:"cql-numDataset"`
	Hidden     []int   `json:"cql-hidden"`
	Alpha      float64 `json:"alpha"`
	Sync       int     `json:"cql-sync"`
	Decay      float64 `json:"cql-decay"`
	Momentum   float64 `json:"cql-momentum"`
	AdamBeta1  float64 `json:"cql-adamBeta1"`
	AdamBeta2  float64 `json:"cql-adamBeta2"`
	AdamEps    float64 `json:"cql-adamEps"`
	L2Lambda   float64 `json:"cql-l2Lambda"`

	Bsize int    `json:"buffer-size"`
	Btype string `json:"buffer-type"`

	StateDim   int  `json:"state-len"`
	BatchSize  int  `json:"cql-batch"`
	IncreaseBS bool `json:"increasing-batch"`

	StateRange []float64 `json:"StateRange"`

	OptName string `json:"optimizer"`

	DataLog         string `json:"datalog"`
	WeightPath      string `json:"weightpath"`
	OfflineLearning bool   `json:"offline-learn"` // during offline learning, output unused action to env
	OnlineLearning  bool   `json:"online-learn"`  // Set to false for offline learning, either true/false for running online.
}

type Cql struct {
	logger.Debug
	rng        *rand.Rand
	lastAction int
	lastState  rlglue.State

	cqlSettings

	updateNum int
	learning  bool
	//stepNum   int

	nml     normalizer.Normalizer
	bf      *buffer.Buffer // Training dataset
	bfValid *buffer.Buffer // Validation dataset

	learningNet network.Network
	targetNet   network.Network
	opt         optimizer.Optimizer

	lw   lockweight.LockWeight
	lock bool
}

func init() {
	Add("cql", NewCql)
}

func NewCql(logger logger.Debug) (rlglue.Agent, error) {
	return &Cql{Debug: logger}, nil
}

func (agent *Cql) InitLockWeight(lw lockweight.LockWeight) lockweight.LockWeight {
	lw.DecCount = 0
	lw.BestAvg = math.Inf(-1)

	if lw.LockCondition == "dynamic-reward" {
		lw.CheckChange = agent.DynamicLock
	} else if lw.LockCondition == "onetime-reward" {
		lw.CheckChange = agent.OnetimeRwdLock
	} else if lw.LockCondition == "onetime-epLength" {
		lw.CheckChange = agent.OnetimeEpLenLock
	} else if lw.LockCondition == "beginning" {
		lw.CheckChange = agent.KeepLock
	}
	return lw
}

func (agent *Cql) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	err := json.Unmarshal(expAttr, &agent.cqlSettings)
	if err != nil {
		return errors.New("CQL agent attributes were not valid: " + err.Error())
	}

	err = json.Unmarshal(expAttr, &agent.lw)
	if err != nil {
		return errors.New("CQL agent LockWeight attributes were not valid: " + err.Error())
	}
	agent.lw = agent.InitLockWeight(agent.lw)

	if agent.DecreasingEpsilon == "None" {
		agent.MinEpsilon = 0 // Not used
	} else {
		agent.Epsilon = 1.0
	}

	agent.learning = false
	//agent.stepNum = 0

	err = json.Unmarshal(envAttr, &agent)

	if err != nil {
		return errors.New("Number of Actions wasn't available: " + err.Error())
	}
	agent.rng = rand.New(rand.NewSource(agent.Seed + int64(run))) // Create a new rand source for reproducibility

	if agent.EnableDebug {
		agent.Message("msg", "agent.Cql Initialize", "seed", agent.Seed, "numberOfActions", agent.NumberOfActions)
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim, agent.Seed+int64(run))
	if agent.cqlSettings.OfflineLearning {
		agent.bfValid = buffer.NewBuffer()
		agent.bfValid.Initialize(agent.Btype, agent.Bsize, agent.StateDim, (agent.Seed+1)%agent.NumDataset+int64(run))
	}

	// Load datalog for offline trainning.
	// To get the trace path, Seed corresponds to run of offline data.
	err = agent.loadDataLog(int(agent.Seed))
	if err != nil {
		return errors.New("Agent failed to load datalog: " + err.Error())
	}

	agent.nml = normalizer.Normalizer{ArrLen: agent.StateDim, ArrRange: agent.StateRange}

	// NN: Graph Construction
	// NN: Weight Initialization
	agent.learningNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.targetNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.updateNum = 0

	// Load neural net for online evaluation/learning.
	err = agent.loadWeights()
	if err != nil {
		return errors.New("Agent failed to load NN weights: " + err.Error())
	}

	if agent.OptName == "Adam" {
		agent.opt = new(optimizer.Adam)
		agent.opt.Init(agent.Alpha, []float64{agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps, agent.L2Lambda}, agent.StateDim, agent.Hidden, agent.NumberOfActions)
	} else if agent.OptName == "Sgd" {
		agent.opt = new(optimizer.Sgd)
		agent.opt.Init(agent.Alpha, []float64{agent.Momentum, agent.L2Lambda}, agent.StateDim, agent.Hidden, agent.NumberOfActions)
	} else {
		return errors.New("Optimizer NotImplemented")
	}

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
//func (agent *Cql) Start(state rlglue.State) rlglue.Action {
func (agent *Cql) Start(oristate rlglue.State) rlglue.Action {
	if agent.cqlSettings.OfflineLearning {
		return rlglue.Action(0)
	}
	state := make([]float64, agent.StateDim)
	copy(state, oristate)

	state = agent.nml.MeanZeroNormalization(state)
	agent.lastState = state
	act := agent.Policy(state)
	agent.lastAction = act

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	return rlglue.Action(act)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
//func (agent *Cql) Step(state rlglue.State, reward float64) rlglue.Action {
func (agent *Cql) Step(oristate rlglue.State, reward float64) rlglue.Action {
	if agent.cqlSettings.OfflineLearning {
		agent.Update()
		return rlglue.Action(0)
	}

	if reward != 0 {
		agent.learning = true
	}
	if agent.DecreasingEpsilon == "step" {
		agent.Epsilon = math.Max(agent.Epsilon-1.0/10000, agent.MinEpsilon)
		fmt.Println(agent.Epsilon)
	}
	state := make([]float64, agent.StateDim)
	copy(state, oristate)
	state = agent.nml.MeanZeroNormalization(state)
	agent.bf.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	//agent.stepNum = agent.stepNum + 1
	agent.Update()
	agent.lastState = state
	agent.lastAction = agent.Policy(state)
	act := rlglue.Action(agent.lastAction)

	if agent.EnableDebug {
		if agent.StateContainsReplay {
			agent.Message("msg", "step", "state", state[0], "reward", reward, "action", act, "expected action", state[1])
		} else {
			agent.Message("msg", "step", "state", state, "reward", reward, "action", act)
		}
	}
	return act
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Cql) End(state rlglue.State, reward float64) {
	if agent.cqlSettings.OfflineLearning {
		agent.Update()
		return
	}
	agent.bf.Feed(agent.lastState, agent.lastAction, state, reward, float64(0)) // gamma=0
	agent.Update()
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Cql) Update() {
	// Deployed online without updating weights.
	if !agent.cqlSettings.OfflineLearning && !agent.cqlSettings.OnlineLearning {
		return
	}

	if !agent.cqlSettings.OfflineLearning && agent.lw.UseLock {
		//if agent.updateNum%agent.Bsize == 0 {
		//	agent.lock = agent.lw.CheckChange()
		//}
		agent.lock = agent.lw.CheckChange()
		if agent.lock {
			agent.updateNum += 1
			return
		}
	}

	if agent.IncreaseBS {
		if agent.updateNum%1000 == 0 {
			//inc := int(math.Min(float64(agent.BatchSize) * agent.Alpha, 1.0))
			inc := int(1 / (1 - float64(agent.AdamBeta1)))
			agent.BatchSize = int(math.Min(float64(agent.BatchSize+inc),
				float64(agent.Bsize/2)))
			fmt.Println("batch size increase to", agent.BatchSize)
		}
	}

	if agent.updateNum%agent.Sync == 0 {
		// NN: Synchronization
		agent.targetNet = network.Synchronization(agent.learningNet, agent.targetNet)
	}

	lastStates, lastActionsFloat, states, rewards, gammas := agent.bf.Sample(agent.BatchSize)
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(lastActionsFloat))

	// NN: Weight update
	lastQ := agent.learningNet.Forward(lastStates)
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQ := agent.targetNet.Predict(states)
	targetActionValue, _ := ao.RowIndexMax(targetQ)

	loss := make([][]float64, len(lastQ))
	for i := 0; i < len(lastQ); i++ {
		loss[i] = make([]float64, agent.NumberOfActions)
	}
	for i := 0; i < len(lastQ); i++ {
		for j := 0; j < agent.NumberOfActions; j++ {
			loss[i][j] = 0
		}
		//loss[i][lastActions[i]] = rewards[i][0] + gammas[i][0]*targetActionValue[i] - lastActionValue[i]
		loss[i][lastActions[i]] = lastActionValue[i] - rewards[i][0] - gammas[i][0]*targetActionValue[i]
	}
	// avgLoss := make([][]float64, 1)
	// avgLoss[0] = make([]float64, agent.NumberOfActions)
	// for j := 0; j < agent.NumberOfActions; j++ {
	// 	sum := 0.0
	// 	for i := 0; i < len(loss); i++ {
	// 		sum += loss[i][j]
	// 	}
	// 	avgLoss[0][j] = sum / float64(len(loss))
	// }

	agent.learningNet.Backward(loss, agent.opt)
	//agent.learningNet.Backward(avgLoss)
	agent.updateNum += 1
}

// Choose action
func (agent *Cql) Policy(state rlglue.State) int {
	var idx int
	if (agent.rng.Float64() < agent.Epsilon) || (!agent.learning) {
		idx = agent.rng.Intn(agent.NumberOfActions)
	} else {
		// NN: choose action
		inputS := make([][]float64, 1)
		inputS[0] = state

		allValue := agent.learningNet.Predict(inputS)
		_, idxs := ao.RowIndexMax(allValue)
		idx = idxs[0]

	}
	return idx
}

func (agent *Cql) CheckAvgRwdLock(avg float64) bool {
	if agent.lw.BestAvg > avg {
		agent.lw.DecCount += 1
		fmt.Println("Count to lock", agent.lw.DecCount)
	} else {
		agent.lw.BestAvg = avg
		agent.lw.DecCount = 0
	}
	var lock bool
	if agent.lw.DecCount > agent.lw.LockThrd {
		agent.lw.DecCount = 0
		lock = true
	} else {
		lock = false
	}
	return lock
}

func (agent *Cql) CheckAvgRwdUnlock(avg float64) bool {
	if agent.lw.LockAvgRwd > avg {
		agent.lw.DecCount += 1
		fmt.Println("Count to unlock", agent.lw.DecCount)
	} else {
		agent.lw.DecCount = 0
	}
	var lock bool
	if agent.lw.DecCount > agent.lw.LockThrd {
		agent.lw.DecCount = 0
		lock = false
	} else {
		lock = true
	}
	return lock
}

//func (agent *Cql) CheckChange() bool {
func (agent *Cql) DynamicLock() bool {
	_, _, _, rewards2D, _ := agent.bf.Content()
	rewards := ao.Flatten2DFloat(rewards2D)
	avg := ao.Average(rewards)
	if len(rewards) < agent.Bsize {
		return false
	}
	if agent.lock {
		lock := agent.CheckAvgRwdUnlock(avg)
		if !lock {
			agent.lw.LockAvgRwd = avg
			agent.lw.DecCount = 0
			fmt.Println("UnLocked")
		}
		return lock
	} else {
		lock := agent.CheckAvgRwdLock(avg)
		if lock {
			agent.lw.LockAvgRwd = avg
			agent.lw.DecCount = 0
			fmt.Println("Locked")
		}
		return lock
	}
}

func (agent *Cql) OnetimeRwdLock() bool {
	if agent.lock {
		return true
	} else {
		_, _, _, rewards2D, _ := agent.bf.Content()
		rewards := ao.Flatten2DFloat(rewards2D)
		avg := ao.Average(rewards)
		if len(rewards) < agent.Bsize {
			return false
		}
		if avg > agent.lw.LockAvgRwd {
			return true
		}
		return false
	}
}

func (agent *Cql) OnetimeEpLenLock() bool {
	if agent.lock {
		return true
	} else {
		_, _, _, rewards2D, _ := agent.bf.Content()
		rewards := ao.Flatten2DFloat(rewards2D)
		if len(rewards) < agent.Bsize {
			return false
		}
		zeros := 0
		for i := 0; i < len(rewards); i++ {
			if rewards[i] == 0 {
				zeros += 1
			}
		}
		if zeros != 0 {
			avg := float64(agent.Bsize) / float64(zeros)
			if avg < agent.lw.LockAvgLen {
				return true
			}
		}
		return false
	}
}

func (agent *Cql) KeepLock() bool {
	return true
}

func (agent *Cql) GetLock() bool {
	return agent.lock
}

// Load datalog, copy dataset to replay buffer.
func (agent *Cql) loadDataLog(run int) error {
	if !agent.cqlSettings.OfflineLearning {
		return nil
	}
	var allTrans [][]float64
	var err error

	// Load training set
	folder := agent.cqlSettings.DataLog
	traceLog := folder + "/traces-" + strconv.Itoa(int(run)) + ".csv"
	fmt.Printf("Training set: %v\n", traceLog)
	allTrans, err = agent.loadDatalogFile(traceLog)
	if err != nil {
		return err
	}

	for i := 0; i < len(allTrans); i++ {
		trans := allTrans[i]
		gamma := float64(0)
		terminal := trans[agent.cqlSettings.StateDim*2+2]
		if terminal == 0 {
			gamma = agent.Gamma
		}
		agent.bf.Feed(
			trans[:agent.cqlSettings.StateDim],                                 // state
			trans[agent.cqlSettings.StateDim],                                  // action
			trans[agent.cqlSettings.StateDim+1:agent.cqlSettings.StateDim*2+1], // next state
			trans[agent.cqlSettings.StateDim*2+1],                              // reward
			gamma,                                                              // gamma
		)
	}

	if !agent.cqlSettings.OfflineLearning {
		return nil
	}

	// Load validation set
	// var allTransValid [][]float64
	traceLog = folder + "/traces-" + strconv.Itoa((run+1)%int(agent.NumDataset)) + ".csv"
	fmt.Printf("Validation set: %v\n", traceLog)
	allTrans, err = agent.loadDatalogFile(traceLog)
	if err != nil {
		return err
	}

	for i := 0; i < len(allTrans); i++ {
		trans := allTrans[i]
		gamma := float64(0)
		terminal := trans[agent.cqlSettings.StateDim*2+2]
		if terminal == 0 {
			gamma = agent.Gamma
		}
		agent.bfValid.Feed(
			trans[:agent.cqlSettings.StateDim],                                 // state
			trans[agent.cqlSettings.StateDim],                                  // action
			trans[agent.cqlSettings.StateDim+1:agent.cqlSettings.StateDim*2+1], // next state
			trans[agent.cqlSettings.StateDim*2+1],                              // reward
			gamma,                                                              // gamma
		)
	}

	return nil
}

func (agent *Cql) loadDatalogFile(tracePath string) ([][]float64, error) {
	// Get offline data
	var allTransTemp [][]float64
	csvFile, err := os.Open(tracePath)
	if err != nil {
		return allTransTemp, errors.New("Cannot find trace log file: " + err.Error())
	}
	allTransStr, err := csv.NewReader(csvFile).ReadAll()
	csvFile.Close()
	if err != nil {
		return allTransTemp, errors.New("Cannot read trace log file: " + err.Error())
	}
	if len(allTransStr) <= agent.cqlSettings.BatchSize {
		return allTransTemp, errors.New("Not enough data to sample from: " + err.Error())
	}
	allTransTemp = make([][]float64, len(allTransStr)-1)
	for i := 1; i < len(allTransStr); i++ { // remove first str (title of column)
		trans := allTransStr[i]
		row := make([]float64, agent.cqlSettings.StateDim*2+3)
		for j, num := range trans {
			if j == 0 { // next state
				num = num[1 : len(num)-1] // remove square brackets
				copy(row[agent.cqlSettings.StateDim+1:agent.cqlSettings.StateDim*2+1], convformat.ListStr2Float(num, " "))
			} else if j == 1 { // current state
				num = num[1 : len(num)-1]
				copy(row[:agent.cqlSettings.StateDim], convformat.ListStr2Float(num, " "))
			} else if j == 2 { // action
				row[agent.cqlSettings.StateDim], _ = strconv.ParseFloat(num, 64)
			} else if j == 3 { //reward
				row[agent.cqlSettings.StateDim*2+1], _ = strconv.ParseFloat(num, 64)
				if row[agent.cqlSettings.StateDim*2+1] == -1 { // termination
					row[agent.cqlSettings.StateDim*2+2] = 1
				} else {
					row[agent.cqlSettings.StateDim*2+2] = 0
				}
			}
		}
		allTransTemp[i-1] = row
	}

	return allTransTemp, nil
}

// Load neural net for online evaluation/learning.
func (agent *Cql) loadWeights() error {
	if agent.cqlSettings.OfflineLearning {
		return nil
	}

	if agent.cqlSettings.WeightPath == "" {
		return nil
	}

	// load weights here, save weights after training (called somewhere in experiment.go)
	err := agent.learningNet.LoadNetwork(
		fmt.Sprintf("%slearning/", agent.cqlSettings.WeightPath),
		agent.StateDim, agent.Hidden, agent.NumberOfActions)
	if err != nil {
		return errors.New("CQL agent unable to load networks: " + err.Error())
	}

	err = agent.targetNet.LoadNetwork(
		fmt.Sprintf("%starget/", agent.cqlSettings.WeightPath),
		agent.StateDim, agent.Hidden, agent.NumberOfActions)
	if err != nil {
		return errors.New("CQL agent unable to load networks: " + err.Error())
	}

	return nil
}

// SaveWeights save neural net weights to speficied path.
func (agent *Cql) SaveWeights(basePath string) error {
	if !agent.cqlSettings.OfflineLearning {
		return nil
	}

	err := agent.learningNet.SaveNetwork(path.Join(agent.cqlSettings.WeightPath, basePath, "learning"))
	if err != nil {
		return errors.New("CQL agent unable to save networks: " + err.Error())
	}

	err = agent.targetNet.SaveNetwork(path.Join(agent.cqlSettings.WeightPath, basePath, "target"))
	if err != nil {
		return errors.New("CQL agent unable to save networks: " + err.Error())
	}

	return nil
}

// GetLearnProg computes mean squared TD error of a full pass over the whole dataset.
func (agent *Cql) GetLearnProg() string {
	// MSTDE of training set
	lastStates, lastActionsFloat, states, rewards, gammas := agent.bf.Content()
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(lastActionsFloat))

	lastQ := agent.learningNet.Predict(lastStates)
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQ := agent.targetNet.Predict(states)
	targetActionValue, _ := ao.RowIndexMax(targetQ)

	loss := 0.0
	for i := 0; i < len(lastQ); i++ {
		diff := rewards[i][0] + gammas[i][0]*targetActionValue[i] - lastActionValue[i]
		loss += math.Pow(diff, 2)
	}

	// MSTDE of validation set
	lastStates, lastActionsFloat, states, rewards, gammas = agent.bfValid.Content()
	lastActions = ao.Flatten2DInt(ao.A64ToInt2D(lastActionsFloat))

	//lastQ = agent.learningNet.Forward(lastStates)
	lastQ = agent.learningNet.Predict(lastStates)
	lastActionValue = ao.RowIndexFloat(lastQ, lastActions)
	//targetQ = agent.targetNet.Predict(states)
	targetQ = agent.learningNet.Predict(states)
	targetActionValue, _ = ao.RowIndexMax(targetQ)

	validLoss := 0.0
	for i := 0; i < len(lastQ); i++ {
		diff := rewards[i][0] + gammas[i][0]*targetActionValue[i] - lastActionValue[i]
		validLoss += math.Pow(diff, 2)
	}

	return fmt.Sprintf("%v,%v",
		strconv.FormatFloat(loss/float64(len(lastQ)), 'f', -1, 64),
		strconv.FormatFloat(validLoss/float64(len(lastQ)), 'f', -1, 64))
}

func (agent *Cql) PassInfo(info string, value float64) interface{} {
	return nil
}
