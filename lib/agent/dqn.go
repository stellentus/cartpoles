package agent

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"

	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/network"
	"gonum.org/v1/gonum/mat"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util/buffer"
)

type Model struct {
}

type Dqn struct {
	logger.Debug
	rng                 *rand.Rand
	lastAction          int
	lastState           rlglue.State
	EnableDebug         bool
	NumberOfActions     int     `json:"numberOfActions"`
	StateContainsReplay bool    `json:"state-contains-replay"`
	Gamma               float64 `json:"gamma"`
	Epsilon             float64 `json:"epsilon"`
	MinEpsilon          float64 `json:"min-epsilon"`
	DecreasingEpsilon   string  `json:"decreasing-epsilon"`
	Hidden              []int   `json:"dqn-hidden"`
	Alpha               float64 `json:"alpha"`
	Sync                int     `json:"dqn-sync"`
	Decay               float64 `json:"dqn-decay"`
	Momentum            float64 `json:"dqn-momentum"`
<<<<<<< HEAD
	AdamBeta1           float64 `json:"dqn-adamBeta1"`
	AdamBeta2           float64 `json:"dqn-adamBeta2"`
	AdamEps             float64 `json:"dqn-adamEps"`

	updateNum int
	learning  bool
=======
	updateNum           int
	learning            bool
	stepNum             int
>>>>>>> 6787cb24067cfbfbe46426f5745f47039997eb70

	bf    *buffer.Buffer
	Bsize int    `json:"buffer-size"`
	Btype string `json:"buffer-type"`

	StateDim  int `json:"state-len"`
	BatchSize int `json:"dqn-batch"`

	StateRange []float64

	learningNet network.Network
	targetNet   network.Network
}

func init() {
	Add("dqn", NewDqn)
}

func NewDqn(logger logger.Debug) (rlglue.Agent, error) {
	return &Dqn{Debug: logger}, nil
}

func (agent *Dqn) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	var ss struct {
		Seed        int64
		EnableDebug bool `json:"enable-debug"`

		NumberOfActions     int     `json:"numberOfActions"`
		StateContainsReplay bool    `json:"state-contains-replay"`
		Gamma               float64 `json:"gamma"`
		Epsilon             float64 `json:"epsilon"`
		MinEpsilon          float64 `json:"min-epsilon"`
		DecreasingEpsilon   string  `json:"decreasing-epsilon"`

		Hidden    []int   `json:"dqn-hidden"`
		Alpha     float64 `json:"alpha"`
		Sync      int     `json:"dqn-sync"`
		Decay     float64 `json:"dqn-decay"`
		Momentum  float64 `json:"dqn-momentum"`
		AdamBeta1 float64 `json:"dqn-adamBeta1"`
		AdamBeta2 float64 `json:"dqn-adamBeta2"`
		AdamEps   float64 `json:"dqn-adamEps"`

		Bsize int    `json:"buffer-size"`
		Btype string `json:"buffer-type"`

		StateDim  int `json:"state-len"`
		BatchSize int `json:"dqn-batch"`

		StateRange []float64 `json:"StateRange"`
	}
	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.Example seed wasn't available: "+err.Error())
		ss.Seed = 0
	}
	agent.EnableDebug = ss.EnableDebug
	agent.NumberOfActions = ss.NumberOfActions
	agent.StateContainsReplay = ss.StateContainsReplay
	agent.Gamma = ss.Gamma

	agent.DecreasingEpsilon = ss.DecreasingEpsilon
	if agent.DecreasingEpsilon == "None" {
		agent.Epsilon = ss.Epsilon
	} else {
		agent.MinEpsilon = ss.MinEpsilon
		agent.Epsilon = 1.0
	}

	agent.Hidden = ss.Hidden
	agent.Alpha = ss.Alpha
	agent.Sync = ss.Sync
	agent.Bsize = ss.Bsize
	agent.Btype = ss.Btype
	agent.StateDim = ss.StateDim
	agent.BatchSize = ss.BatchSize
	agent.StateRange = ss.StateRange
	agent.Decay = ss.Decay
	agent.Momentum = ss.Momentum
	agent.AdamBeta1 = ss.AdamBeta1
	agent.AdamBeta2 = ss.AdamBeta2
	agent.AdamEps = ss.AdamEps
	agent.learning = false
	agent.stepNum = 0

	err = json.Unmarshal(envAttr, &agent)

	// fmt.Println(agent.Gamma)
	// fmt.Println(agent.Epsilon)
	// fmt.Println(agent.DecreasingEpsilon)
	// fmt.Println(agent.Hidden)
	// fmt.Println(agent.Alpha)
	// fmt.Println(agent.Sync)
	// fmt.Println(agent.Bsize)
	// fmt.Println(agent.Btype)
	// fmt.Println(agent.StateDim)
	// fmt.Println(agent.BatchSize)
	// fmt.Println(agent.StateRange)
	// fmt.Println(agent.learning)
	// fmt.Println(agent.learning)
	// fmt.Println(agent.Momentum)

	// os.Exit(-1)

	if err != nil {
		agent.Message("err", "agent.Example number of Actions wasn't available: "+err.Error())
	}
	// agent.rng = rand.New(rand.NewSource(ss.Seed)) // Create a new rand source for reproducibility
	// agent.lastAction = rng.Intn(agent.NumberOfActions)
	agent.rng = rand.New(rand.NewSource(ss.Seed + int64(run))) // Create a new rand source for reproducibility

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", ss.Seed, "numberOfActions", agent.NumberOfActions)
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim, ss.Seed+int64(run))

	// NN: Graph Construction
	// NN: Weight Initialization
	agent.learningNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta1, agent.AdamEps)
	agent.targetNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta1, agent.AdamEps)
	agent.updateNum = 0

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Dqn) Start(state rlglue.State) rlglue.Action {
	// if agent.DecreasingEpsilon == "ep" {
	// 	agent.Epsilon = math.Max(agent.Epsilon-0.05, agent.MinEpsilon)
	// }

	state = agent.StateNormalization(state)
	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	agent.lastState = state
	act := agent.Policy(state)
	agent.lastAction = act
	return rlglue.Action(act)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Dqn) Step(state rlglue.State, reward float64) rlglue.Action {
	if reward != 0 {
		agent.learning = true
	}
	if agent.DecreasingEpsilon == "step" {
		agent.Epsilon = math.Max(agent.Epsilon-1.0/10000, agent.MinEpsilon)
		fmt.Println(agent.Epsilon)
	}

	state = agent.StateNormalization(state)
	agent.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	agent.stepNum = agent.stepNum + 1
	agent.Update()
	agent.lastAction = agent.Policy(state)
	agent.lastState = state

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
func (agent *Dqn) End(state rlglue.State, reward float64) {
	agent.Feed(agent.lastState, agent.lastAction, state, reward, float64(0))
	// agent.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Dqn) StateNormalization(state rlglue.State) rlglue.State {
	for i := 0; i < agent.StateDim; i++ {
		state[i] = state[i] / agent.StateRange[i]
	}
	return state
}

func (agent *Dqn) Feed(lastS rlglue.State, lastA int, state rlglue.State, reward float64, gamma float64) {
	agent.bf.Feed(lastS, lastA, state, reward, gamma)
}

func (agent *Dqn) Update() {

	if agent.updateNum%agent.Sync == 0 {
		// NN: Synchronization
		// for i := 0; i < len(agent.targetNet.HiddenWeights); i++ {
		// 	fmt.Println(agent.learningNet.HiddenWeights[i])
		// 	fmt.Println(agent.targetNet.HiddenWeights[i])
		// 	fmt.Println("---")
		// }
		// fmt.Println(agent.learningNet.OutputWeights)
		// fmt.Println(agent.targetNet.OutputWeights)
		// fmt.Println("===========")

		for i := 0; i < len(agent.targetNet.HiddenWeights); i++ {
			agent.targetNet.HiddenWeights[i] = agent.learningNet.HiddenWeights[i]
		}
		agent.targetNet.OutputWeights = agent.learningNet.OutputWeights

		// for i := 0; i < len(agent.targetNet.HiddenWeights); i++ {
		// 	fmt.Println(agent.learningNet.HiddenWeights[i])
		// 	fmt.Println(agent.targetNet.HiddenWeights[i])
		// 	fmt.Println("---")
		// }
		// fmt.Println(agent.learningNet.OutputWeights)
		// fmt.Println(agent.targetNet.OutputWeights)

		// fmt.Println("Sync")
		// os.Exit(-1)
	}

	if agent.stepNum < agent.BatchSize {
		return
	}

	samples := agent.bf.Sample(agent.BatchSize)
	lastStates := ao.Index2d(samples, 0, len(samples), 0, agent.StateDim)
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(ao.Index2d(samples, 0, len(samples), agent.StateDim, agent.StateDim+1)))
	states := ao.Index2d(samples, 0, len(samples), agent.StateDim+1, agent.StateDim*2+1)
	rewards := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), agent.StateDim*2+1, agent.StateDim*2+2))
	gammas := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), agent.StateDim*2+2, agent.StateDim*2+3))

	// NN: Weight update
	// fmt.Println("Input")
	// fmt.Println(lastStates)
	lastQMat := agent.learningNet.Forward(lastStates)
	var lastQ [][]float64
	for i := 0; i < len(lastStates); i++ {
		lastQ = append(lastQ, mat.Row(nil, i, lastQMat))
	}
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQMat := agent.learningNet.Predict(states)
	var targetQ [][]float64
	for i := 0; i < len(states); i++ {
		targetQ = append(targetQ, mat.Row(nil, i, targetQMat))
	}
	targetActionValue := ao.RowIndexMax(targetQ)

	// loss := float64(0)
	// for i := 0; i < len(lastQ); i++ {
	// 	// loss = loss + math.Pow(rewards[i]+gammas[i]*targetActionValue[i]-lastActionValue[i], 2)
	// 	loss = loss + rewards[i] + gammas[i]*targetActionValue[i] - lastActionValue[i]
	// }
	// loss = loss / float64(len(lastQ))
	// fmt.Println(loss)

	loss := make([][]float64, len(lastQ))
	for i := 0; i < len(lastQ); i++ {
		loss[i] = make([]float64, agent.NumberOfActions)
	}
	for i := 0; i < len(lastQ); i++ {
		for j := 0; j < agent.NumberOfActions; j++ {
			loss[i][j] = 0
		}
		// loss[i][lastActions[i]] = math.Pow(rewards[i]+gammas[i]*targetActionValue[i]-lastActionValue[i], 2)
		// loss[i][lastActions[i]] = math.Sqrt(loss[i][lastActions[i]] / 2.0)
		loss[i][lastActions[i]] = rewards[i] + gammas[i]*targetActionValue[i] - lastActionValue[i]
	}
	// fmt.Println(loss[0][lastActions[0]])

	agent.learningNet.Backward(loss)
	agent.updateNum += 1
}

// Choose action
func (agent *Dqn) Policy(state rlglue.State) int {
	var idx int
	if (agent.rng.Float64() < agent.Epsilon) || (!agent.learning) {
		idx = agent.rng.Intn(agent.NumberOfActions)
	} else {
		// NN: choose action
		inputS := make([][]float64, 1)
		inputS[0] = agent.lastState
		allValueMat := agent.learningNet.Predict(inputS)
		allValue := mat.Row(nil, 0, allValueMat)
		var argmax int
		var v float64
		max := math.Inf(-1)
		for i := 0; i < agent.NumberOfActions; i++ {
			v = allValue[i]
			if v > max {
				max = v
				argmax = i
			}
		}
		// fmt.Println(allValue, argmax)
		idx = argmax
	}
	return idx
}
