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
	"github.com/stellentus/cartpoles/lib/util"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/buffer"
	"github.com/stellentus/cartpoles/lib/util/convformat"
	"github.com/stellentus/cartpoles/lib/util/lockweight"
	"github.com/stellentus/cartpoles/lib/util/network"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
)

type cqlLinearSettings struct {
	Seed        int64
	EnableDebug bool `json:"enable-debug"`

	NumTilings          int     `json:"tilings"`
	NumTiles            int     `json:"tiles"`
	EnvName             string  `json:"env-name"`
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

	// StateRange []float64 `json:"StateRange"`

	OptName string `json:"optimizer"`

	DataLog         string `json:"datalog"`
	WeightPath      string `json:"weightpath"`
	OfflineLearning bool   `json:"offline-learn"` // during offline learning, output unused action to env
	OnlineLearning  bool   `json:"online-learn"`  // Set to false for offline learning, either true/false for running online.
}

type CqlLinear struct {
	logger.Debug
	rng                *rand.Rand
	tiler              util.MultiTiler
	tilerNumIndices    int
	lastAction         int
	lastState          rlglue.State
	lastTileCodedState rlglue.State

	cqlLinearSettings

	updateNum int
	learning  bool
	//stepNum   int

	stateRange [][]float64
	// nml normalizer.Normalizer
	bf      *buffer.Buffer // Training dataset
	bfValid *buffer.Buffer // Validation dataset

	learningNet network.Network
	targetNet   network.Network
	opt         optimizer.Optimizer

	lw   lockweight.LockWeight
	lock bool
}

func init() {
	Add("cqllinear", NewCqlLinear)
}

func NewCqlLinear(logger logger.Debug) (rlglue.Agent, error) {
	return &CqlLinear{Debug: logger}, nil
}

func (agent *CqlLinear) InitLockWeight(lw lockweight.LockWeight) lockweight.LockWeight {
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

func (agent *CqlLinear) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	err := json.Unmarshal(expAttr, &agent.cqlLinearSettings)
	if err != nil {
		return errors.New("CQLLinear agent attributes were not valid: " + err.Error())
	}

	err = json.Unmarshal(expAttr, &agent.lw)
	if err != nil {
		return errors.New("CQLLinear agent LockWeight attributes were not valid: " + err.Error())
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

	// Initialize tiler
	err = agent.InitTiler()
	if err != nil {
		return errors.New("Failed to initialize tilder: " + err.Error())
	}

	if agent.EnableDebug {
		agent.Message("msg", "agent.CqlLinear Initialize", "seed", agent.Seed, "numberOfActions", agent.NumberOfActions)
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.tilerNumIndices, agent.Seed+int64(run))
	if agent.cqlLinearSettings.OfflineLearning {
		agent.bfValid = buffer.NewBuffer()
		agent.bfValid.Initialize(agent.Btype, agent.Bsize, agent.tilerNumIndices, (agent.Seed+1)%agent.NumDataset+int64(run))
	}

	// Load datalog for offline trainning.
	// To get the trace path, Seed corresponds to run of offline data.
	err = agent.loadDataLog(int(agent.Seed))
	if err != nil {
		return errors.New("Agent failed to load datalog: " + err.Error())
	}

	// agent.nml = normalizer.Normalizer{agent.StateDim, agent.StateRange}

	// NN: Graph Construction
	// NN: Weight Initialization
	agent.learningNet = network.CreateNetwork(
		agent.tilerNumIndices, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.targetNet = network.CreateNetwork(
		agent.tilerNumIndices, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.updateNum = 0

	// Load neural net for online evaluation/learning.
	err = agent.loadWeights()
	if err != nil {
		return errors.New("Agent failed to load NN weights: " + err.Error())
	}

	if agent.OptName == "Adam" {
		agent.opt = new(optimizer.Adam)
		agent.opt.Init(
			agent.Alpha, []float64{agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps, agent.L2Lambda},
			agent.tilerNumIndices, agent.Hidden, agent.NumberOfActions)
	} else if agent.OptName == "Sgd" {
		agent.opt = new(optimizer.Sgd)
		agent.opt.Init(
			agent.Alpha, []float64{agent.Momentum, agent.L2Lambda},
			agent.tilerNumIndices, agent.Hidden, agent.NumberOfActions)
	} else {
		return errors.New("Optimizer NotImplemented")
	}

	return nil
}

func (agent *CqlLinear) InitTiler() error {
	// scales the input observations for tile-coding
	var err error
	if agent.cqlLinearSettings.EnvName == "cartpole" {
		agent.cqlLinearSettings.NumberOfActions = 2
		scalers := []util.Scaler{
			util.NewScaler(-maxPosition, maxPosition, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxVelocity, maxVelocity, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxAngle, maxAngle, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxAngularVelocity, maxAngularVelocity, agent.cqlLinearSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(4, agent.cqlLinearSettings.NumTilings, scalers)
		if err != nil {
			return err
		}

		agent.stateRange = make([][]float64, 4)
		agent.stateRange[0] = make([]float64, 2)
		agent.stateRange[0][0] = -maxPosition
		agent.stateRange[0][1] = maxPosition
		agent.stateRange[1] = make([]float64, 2)
		agent.stateRange[1][0] = -maxVelocity
		agent.stateRange[1][1] = maxVelocity
		agent.stateRange[2] = make([]float64, 2)
		agent.stateRange[2][0] = -maxAngle
		agent.stateRange[2][1] = maxAngle
		agent.stateRange[3] = make([]float64, 2)
		agent.stateRange[3][0] = -maxAngularVelocity
		agent.stateRange[3][1] = maxAngularVelocity

	} else if agent.cqlLinearSettings.EnvName == "acrobot" {
		agent.cqlLinearSettings.NumberOfActions = 2 //3
		scalers := []util.Scaler{
			util.NewScaler(-maxFeature1, maxFeature1, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxFeature2, maxFeature2, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxFeature3, maxFeature3, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxFeature4, maxFeature4, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxFeature5, maxFeature5, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(-maxFeature6, maxFeature6, agent.cqlLinearSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(6, agent.cqlLinearSettings.NumTilings, scalers)
		if err != nil {
			return err
		}

		agent.stateRange = make([][]float64, 6)
		agent.stateRange[0] = make([]float64, 2)
		agent.stateRange[0][0] = -maxFeature1
		agent.stateRange[0][1] = maxFeature1
		agent.stateRange[1] = make([]float64, 2)
		agent.stateRange[1][0] = -maxFeature2
		agent.stateRange[1][1] = maxFeature2
		agent.stateRange[2] = make([]float64, 2)
		agent.stateRange[2][0] = -maxFeature3
		agent.stateRange[2][1] = maxFeature3
		agent.stateRange[3] = make([]float64, 2)
		agent.stateRange[3][0] = -maxFeature4
		agent.stateRange[3][1] = maxFeature4
		agent.stateRange[4] = make([]float64, 2)
		agent.stateRange[4][0] = -maxFeature5
		agent.stateRange[4][1] = maxFeature5
		agent.stateRange[5] = make([]float64, 2)
		agent.stateRange[5][0] = -maxFeature6
		agent.stateRange[5][1] = maxFeature6

	} else if agent.cqlLinearSettings.EnvName == "puddleworld" {
		agent.cqlLinearSettings.NumberOfActions = 4 // 5
		scalers := []util.Scaler{
			util.NewScaler(minFeature1, maxFeature1, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(minFeature2, maxFeature2, agent.cqlLinearSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(2, agent.cqlLinearSettings.NumTilings, scalers)
		if err != nil {
			return err
		}

		agent.stateRange = make([][]float64, 2)
		agent.stateRange[0] = make([]float64, 2)
		agent.stateRange[0][0] = minFeature1
		agent.stateRange[0][1] = maxFeature1
		agent.stateRange[1] = make([]float64, 2)
		agent.stateRange[1][0] = minFeature2
		agent.stateRange[1][1] = maxFeature2

	} else if agent.cqlLinearSettings.EnvName == "gridworld" {
		agent.cqlLinearSettings.NumberOfActions = 4 // 5
		scalers := []util.Scaler{
			util.NewScaler(minCoord, maxCoord, agent.cqlLinearSettings.NumTiles),
			util.NewScaler(minCoord, maxCoord, agent.cqlLinearSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(2, agent.cqlLinearSettings.NumTilings, scalers)
		if err != nil {
			return err
		}

		agent.stateRange = make([][]float64, 2)
		agent.stateRange[0] = make([]float64, 2)
		agent.stateRange[0][0] = minCoord
		agent.stateRange[0][1] = maxCoord
		agent.stateRange[1] = make([]float64, 2)
		agent.stateRange[1][0] = minCoord
		agent.stateRange[1][1] = maxCoord

	} else {
		return errors.New("Environment NotImplemented")
	}

	agent.FillHashTable()

	agent.tilerNumIndices = agent.tiler.NumberOfIndices()

	return nil
}

func (agent *CqlLinear) FillHashTable() {
	//var fixedCoord []float64
	//fmt.Println(agent.generateAllPoints(fixedCoord, 0, 0))

	//fmt.Println(agent.generateAllPoints())
	agent.generateAllPoints()
}

func (agent *CqlLinear) generateAllPoints() int {
	tempS := make([]float64, agent.StateDim)
	count := 0
	for dim := 0; dim < agent.StateDim; dim++ {
		maxRange := agent.stateRange[dim][1] - agent.stateRange[dim][0]
		minS := agent.stateRange[dim][0]
		numBlock := agent.NumTilings * agent.NumTiles
		blockLen := maxRange / float64(numBlock)
		for k := 0; k < agent.StateDim; k++ {
			tempS[k] = 0
		}
		for i := 0; i < numBlock+1; i++ {
			tempS[dim] = math.Max(math.Min(minS+float64(i)*blockLen, agent.stateRange[dim][1]), agent.stateRange[dim][0])
			agent.tiler.Tile(tempS)
			count += 1
		}
	}
	for dim := 0; dim < agent.StateDim; dim++ {
		maxRange0 := agent.stateRange[dim][1] - agent.stateRange[dim][0]
		for pair := dim + 1; pair < agent.StateDim; pair++ {
			maxRange1 := agent.stateRange[pair][1] - agent.stateRange[pair][0]
			minS0 := agent.stateRange[dim][0]
			minS1 := agent.stateRange[pair][0]
			numBlock := agent.NumTilings * agent.NumTiles
			blockLen0 := maxRange0 / float64(numBlock)
			blockLen1 := maxRange1 / float64(numBlock)
			for i := 0; i < numBlock+1; i++ {
				for j := 0; j < numBlock+1; j++ {

					for k := 0; k < agent.StateDim; k++ {
						tempS[k] = 0
					}
					tempS[dim] = math.Max(math.Min(minS0+float64(i)*blockLen0, agent.stateRange[dim][1]), agent.stateRange[dim][0])
					tempS[pair] = math.Max(math.Min(minS1+float64(j)*blockLen1, agent.stateRange[dim][1]), agent.stateRange[dim][0])
					agent.tiler.Tile(tempS)

					//fmt.Println(tempS[dim], minS0 + float64(i)*blockLen0,  tempS[pair], minS1 + float64(j)*blockLen1)
					count += 1
				}
			}
		}
	}
	return count
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *CqlLinear) Start(oristate rlglue.State) rlglue.Action {
	if agent.cqlLinearSettings.OfflineLearning {
		return rlglue.Action(0)
	}

	state := make([]float64, agent.StateDim)
	copy(state, oristate)
	agent.lastState = state
	agent.lastTileCodedState = agent.tileEncodeState(oristate)
	act := agent.Policy(agent.lastTileCodedState)
	agent.lastAction = act

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	return rlglue.Action(act)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
//func (agent *CqlLinear) Step(state rlglue.State, reward float64) rlglue.Action {
func (agent *CqlLinear) Step(oristate rlglue.State, reward float64) rlglue.Action {
	if agent.cqlLinearSettings.OfflineLearning {
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

	tileCodedState := agent.tileEncodeState(oristate)
	// state = agent.nml.MeanZeroNormalization(state)
	agent.bf.Feed(agent.lastTileCodedState, agent.lastAction, tileCodedState, reward, agent.Gamma)
	//agent.stepNum = agent.stepNum + 1
	agent.Update()
	agent.lastState = state
	agent.lastTileCodedState = tileCodedState
	agent.lastAction = agent.Policy(tileCodedState)
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
func (agent *CqlLinear) End(state rlglue.State, reward float64) {
	if agent.cqlLinearSettings.OfflineLearning {
		agent.Update()
	}
	tileCodedState := agent.tileEncodeState(state)
	agent.bf.Feed(agent.lastTileCodedState, agent.lastAction, tileCodedState, reward, float64(0)) // gamma=0
	agent.Update()
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *CqlLinear) Update() {
	// Deployed online without updating weights.
	if !agent.cqlLinearSettings.OfflineLearning && !agent.cqlLinearSettings.OnlineLearning {
		return
	}

	if !agent.cqlLinearSettings.OfflineLearning && agent.lw.UseLock {
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
		//for i := 0; i < len(agent.targetNet.HiddenWeights); i++ {
		//	agent.targetNet.HiddenWeights[i] = agent.learningNet.HiddenWeights[i]
		//}
		//agent.targetNet.OutputWeights = agent.learningNet.OutputWeights
		////fmt.Println("sync", agent.updateNum)
		agent.targetNet = network.Synchronization(agent.learningNet, agent.targetNet)
	}
	lastTileCodedStates, lastActionsFloat, tileCodedStates, rewards, gammas := agent.bf.Sample(agent.BatchSize)
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(lastActionsFloat))

	// lastStatesTileCoded := agent.tileEncodeStates(lastStates)
	// statesTileCoded := agent.tileEncodeStates(states)
	// NN: Weight update
	lastQ := agent.learningNet.Forward(lastTileCodedStates)
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQ := agent.targetNet.Predict(tileCodedStates)
	targetActionValue, _ := ao.RowIndexMax(targetQ)

	loss := make([][]float64, len(lastQ))
	for i := 0; i < len(lastQ); i++ {
		loss[i] = make([]float64, agent.NumberOfActions)
	}
	for i := 0; i < len(lastQ); i++ {
		for j := 0; j < agent.NumberOfActions; j++ {
			loss[i][j] = 0
		}
		loss[i][lastActions[i]] = lastActionValue[i] - rewards[i][0] - gammas[i][0]*targetActionValue[i]
	}
	constr := make([]float64, len(lastQ))
	for i := 0; i < len(lastQ); i++ {
		constr[i] = ao.LogSumExp(lastQ[i]) - lastActionValue[i]
	}
	avgConstr := ao.Average(constr) - ao.Average(lastActionValue)

	avgLossOne := make([][]float64, 1)
	avgLossOne[0] = make([]float64, agent.NumberOfActions)
	for j := 0; j < agent.NumberOfActions; j++ {
		sum := 0.0
		for i := 0; i < len(loss); i++ {
			sum += loss[i][j]
		}
		avgLossOne[0][j] = sum / float64(len(loss)) * 0.5 + avgConstr
	}
	avgLoss := make([][]float64, len(loss))
	for i := 0; i < len(avgLoss); i++ {
		avgLoss[i] = avgLossOne[0]
	}

	//agent.learningNet.Backward(loss, agent.opt)
	agent.learningNet.Backward(avgLoss, agent.opt)
	agent.updateNum += 1
}

// Choose action
func (agent *CqlLinear) Policy(tileCodedState rlglue.State) int {
	var idx int
	if (agent.rng.Float64() < agent.Epsilon) || (!agent.learning) {
		idx = agent.rng.Intn(agent.NumberOfActions)
	} else {
		// NN: choose action
		inputS := make([][]float64, 1)
		inputS[0] = tileCodedState

		allValue := agent.learningNet.Predict(inputS)
		_, idxs := ao.RowIndexMax(allValue)
		idx = idxs[0]

	}
	return idx
}

func (agent *CqlLinear) tileEncodeStates(rawStates [][]float64) [][]float64 {
	tileCodedStates := make([][]float64, len(rawStates))
	for i := 0; i < len(rawStates); i++ {
		tileCodedStates[i] = agent.tileEncodeState(rawStates[i])
	}
	return tileCodedStates
}

func (agent *CqlLinear) tileEncodeState(rawState []float64) []float64 {
	stateActiveFeatures, err := agent.tiler.Tile(rawState) // Indices of active features of the tile-coded state
	if err != nil {
		agent.Message("err", "agent.CQLLinear is acting on garbage state because it couldn't create tiles: "+err.Error())
	}
	state := make(rlglue.State, agent.tilerNumIndices)
	for _, v := range stateActiveFeatures {
		state[v] = 1.0
	}
	return state
}

func (agent *CqlLinear) CheckAvgRwdLock(avg float64) bool {
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

func (agent *CqlLinear) CheckAvgRwdUnlock(avg float64) bool {
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

//func (agent *CqlLinear) CheckChange() bool {
func (agent *CqlLinear) DynamicLock() bool {
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

func (agent *CqlLinear) OnetimeRwdLock() bool {
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

func (agent *CqlLinear) OnetimeEpLenLock() bool {
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

func (agent *CqlLinear) KeepLock() bool {
	return true
}

func (agent *CqlLinear) GetLock() bool {
	return agent.lock
}

// Load datalog, copy dataset to replay buffer.
func (agent *CqlLinear) loadDataLog(run int) error {
	if !agent.cqlLinearSettings.OfflineLearning {
		return nil
	}
	var allTrans [][]float64
	var err error

	// Load training set
	folder := agent.cqlLinearSettings.DataLog
	traceLog := folder + "/traces-" + strconv.Itoa(int(run)) + ".csv"
	fmt.Printf("Training set: %v\n", traceLog)
	allTrans, err = agent.loadDatalogFile(traceLog)
	if err != nil {
		return err
	}

	for i := 0; i < len(allTrans); i++ {
		trans := allTrans[i]
		gamma := float64(0)
		terminal := trans[agent.cqlLinearSettings.StateDim*2+2]
		if terminal == 0 {
			gamma = agent.Gamma
		}
		agent.bf.Feed(
			agent.tileEncodeState(trans[:agent.cqlLinearSettings.StateDim]),                                       // state
			trans[agent.cqlLinearSettings.StateDim],                                        // action
			agent.tileEncodeState(trans[agent.cqlLinearSettings.StateDim+1:agent.cqlLinearSettings.StateDim*2+1]), // next state
			trans[agent.cqlLinearSettings.StateDim*2+1],                                    // reward
			gamma, // gamma
		)
	}

	if !agent.cqlLinearSettings.OfflineLearning {
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
		terminal := trans[agent.cqlLinearSettings.StateDim*2+2]
		if terminal == 0 {
			gamma = agent.Gamma
		}
		agent.bfValid.Feed(
			agent.tileEncodeState(trans[:agent.cqlLinearSettings.StateDim]),                                       // state
			trans[agent.cqlLinearSettings.StateDim],                                        // action
			agent.tileEncodeState(trans[agent.cqlLinearSettings.StateDim+1:agent.cqlLinearSettings.StateDim*2+1]), // next state
			trans[agent.cqlLinearSettings.StateDim*2+1],                                    // reward
			gamma, // gamma
		)
	}

	return nil
}

func (agent *CqlLinear) loadDatalogFile(tracePath string) ([][]float64, error) {
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
	if len(allTransStr) <= agent.cqlLinearSettings.BatchSize {
		return allTransTemp, errors.New("Not enough data to sample from: " + err.Error())
	}
	allTransTemp = make([][]float64, len(allTransStr)-1)
	for i := 1; i < len(allTransStr); i++ { // remove first str (title of column)
		trans := allTransStr[i]
		row := make([]float64, agent.cqlLinearSettings.StateDim*2+3)
		for j, num := range trans {
			if j == 0 { // next state
				num = num[1 : len(num)-1] // remove square brackets
				copy(row[agent.cqlLinearSettings.StateDim+1:agent.cqlLinearSettings.StateDim*2+1], convformat.ListStr2Float(num, " "))
			} else if j == 1 { // current state
				num = num[1 : len(num)-1]
				copy(row[:agent.cqlLinearSettings.StateDim], convformat.ListStr2Float(num, " "))
			} else if j == 2 { // action
				row[agent.cqlLinearSettings.StateDim], _ = strconv.ParseFloat(num, 64)
			} else if j == 3 { //reward
				row[agent.cqlLinearSettings.StateDim*2+1], _ = strconv.ParseFloat(num, 64)
				if row[agent.cqlLinearSettings.StateDim*2+1] == -1 { // termination
					row[agent.cqlLinearSettings.StateDim*2+2] = 1
				} else {
					row[agent.cqlLinearSettings.StateDim*2+2] = 0
				}
			}
		}
		allTransTemp[i-1] = row
	}

	return allTransTemp, nil
}

// Load neural net for online evaluation/learning.
func (agent *CqlLinear) loadWeights() error {
	if agent.cqlLinearSettings.OfflineLearning {
		return nil
	}

	if agent.cqlLinearSettings.WeightPath == "" {
		return nil
	}

	// load weights here, save weights after training (called somewhere in experiment.go)
	err := agent.learningNet.LoadNetwork(
		fmt.Sprintf("%slearning/", agent.cqlLinearSettings.WeightPath),
		agent.tilerNumIndices, agent.Hidden, agent.NumberOfActions)
	if err != nil {
		return errors.New("CQLLinear agent unable to load networks: " + err.Error())
	}

	err = agent.targetNet.LoadNetwork(
		fmt.Sprintf("%starget/", agent.cqlLinearSettings.WeightPath),
		agent.tilerNumIndices, agent.Hidden, agent.NumberOfActions)
	if err != nil {
		return errors.New("CQLLinear agent unable to load networks: " + err.Error())
	}

	return nil
}

// SaveWeights save neural net weights to speficied path.
func (agent *CqlLinear) SaveWeights(basePath string) error {
	if !agent.cqlLinearSettings.OfflineLearning {
		return nil
	}

	err := agent.learningNet.SaveNetwork(path.Join(agent.cqlLinearSettings.WeightPath, basePath, "learning"))
	if err != nil {
		return errors.New("CQLLinear agent unable to save networks: " + err.Error())
	}

	err = agent.targetNet.SaveNetwork(path.Join(agent.cqlLinearSettings.WeightPath, basePath, "target"))
	if err != nil {
		return errors.New("CQLLinear agent unable to save networks: " + err.Error())
	}

	return nil
}

// GetLearnProg computes mean squared TD error of a full pass over the whole dataset.
func (agent *CqlLinear) GetLearnProg() string {
	// MSTDE of training set
	lastStates, lastActionsFloat, states, rewards, gammas := agent.bf.Content()
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(lastActionsFloat))
	//fmt.Println(lastStates)
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

func (agent *CqlLinear) PassInfo(info string, value float64) interface{} {
	return nil
}