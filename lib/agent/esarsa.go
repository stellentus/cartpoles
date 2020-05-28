package agent

import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util"
)

const (
	maxPosition        = 2.4
	maxVelocity        = 4
	maxAngle           = 12 * 2 * math.Pi / 360
	maxAngularVelocity = 3.5
)

// Expected sarsa-lambda with tile coding
type ESarsa struct {
	logger.Debug
	rng                 *rand.Rand
	tiler               util.MultiTiler
	EnableDebug         bool
	StateContainsReplay bool

	// Agent accessible parameters
	weights                [][]float64 // weights is a slice of weights for each action
	traces                 [][]float64
	delta                  float64
	oldStateActiveFeatures []int
	oldAction              rlglue.Action
	gamma                  float64
	lambda                 float64
	epsilon                float64
	alpha                  float64
	stepsize               float64
}

func init() {
	Add("esarsa", NewESarsa)
}

func NewESarsa(logger logger.Debug) (rlglue.Agent, error) {
	return &ESarsa{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *ESarsa) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	ss := struct {
		// Experiment parameters from config
		EnableDebug         bool    `json:"enable-debug"`
		StateContainsReplay bool    `json:"state-contains-replay"`
		Seed                int64   `json:"seed"`
		NumTilings          int     `json:"tilings"`
		NumTiles            int     `json:"tiles"`
		Gamma               float64 `json:"gamma"`
		Lambda              float64 `json:"lambda"`
		Epsilon             float64 `json:"epsilon"`
		Alpha               float64 `json:"alpha"`
	}{
		// These default settings will be used if the config doesn't set these values
		EnableDebug:         false,
		StateContainsReplay: false,
		Seed:                0,
		NumTilings:          32,
		NumTiles:            4,
		Gamma:               0.99,
		Lambda:              0.8,
		Epsilon:             0.05,
		Alpha:               0.1,
	}

	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.ESarsa seed wasn't available: "+err.Error())
		ss.Seed = 0
	}

	agent.EnableDebug = ss.EnableDebug
	agent.StateContainsReplay = ss.StateContainsReplay
	agent.gamma = ss.Gamma
	agent.lambda = ss.Lambda
	agent.epsilon = ss.Epsilon
	agent.alpha = ss.Alpha
	agent.stepsize = ss.Alpha / float64(ss.NumTilings)         // Setting stepsize
	agent.rng = rand.New(rand.NewSource(ss.Seed + int64(run))) // Create a new rand source for reproducibility

	// scales the input observations for tile-coding
	scalers := []util.Scaler{
		util.NewScaler(-maxPosition, maxPosition, ss.NumTiles),
		util.NewScaler(-maxVelocity, maxVelocity, ss.NumTiles),
		util.NewScaler(-maxAngle, maxAngle, ss.NumTiles),
		util.NewScaler(-maxAngularVelocity, maxAngularVelocity, ss.NumTiles),
	}

	agent.tiler, err = util.NewMultiTiler(4, ss.NumTilings, scalers)
	if err != nil {
		return err
	}

	agent.weights = make([][]float64, 2) // one weight slice for each action
	agent.weights[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.weights[1] = make([]float64, agent.tiler.NumberOfIndices())

	agent.traces = make([][]float64, 2) // one trace slice for each action
	agent.traces[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.traces[1] = make([]float64, agent.tiler.NumberOfIndices())

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *ESarsa) Start(state rlglue.State) rlglue.Action {
	var err error
	agent.oldStateActiveFeatures, err = agent.tiler.Tile(state) // Indices of active features of the tile-coded state

	if err != nil {
		agent.Message("err", "agent.ESarsa is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	agent.oldAction, _ = agent.PolicyExpectedSarsaLambda(agent.oldStateActiveFeatures) // Exp-Sarsa-L policy

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}

	return agent.oldAction

}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *ESarsa) Step(state rlglue.State, reward float64) rlglue.Action {
	newStateActiveFeatures, err := agent.tiler.Tile(state) // Indices of active features of the tile-coded state

	if err != nil {
		agent.Message("err", "agent.ESarsa is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	agent.delta = reward // TD error calculation begins

	for _, value := range agent.oldStateActiveFeatures {
		agent.delta -= agent.weights[agent.oldAction][value] // TD error prediction calculation
		agent.traces[agent.oldAction][value] = 1             // replacing active traces to 1
	}

	newAction, epsilons := agent.PolicyExpectedSarsaLambda(newStateActiveFeatures) // Exp-Sarsa-L policy

	for j := range []int{0, 1} {
		for _, value := range newStateActiveFeatures {
			agent.delta += agent.gamma * epsilons[j] * agent.weights[j][value] // TD error target calculation
		}
	}

	// update for both actions for weights and traces
	for j := range []int{0, 1} {
		for i := range agent.weights[j] {
			if agent.traces[j][i] != 0 { // update only where traces are non-zero
				agent.weights[j][i] += agent.stepsize * agent.delta * agent.traces[j][i] // Semi-gradient descent, update weights
				agent.traces[j][i] = agent.gamma * agent.lambda * agent.traces[j][i]     // update traces
			}
		}
	}

	// New information is old for the next time step
	agent.oldStateActiveFeatures = newStateActiveFeatures
	agent.oldAction = newAction

	if agent.EnableDebug {
		if agent.StateContainsReplay {
			agent.Message("msg", "step", "state", state[0], "reward", reward, "action", agent.oldAction, "expected action", agent.oldAction)
		} else {
			agent.Message("msg", "step", "state", state, "reward", reward, "action", agent.oldAction)
		}
	}

	return agent.oldAction
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *ESarsa) End(state rlglue.State, reward float64) {
	agent.Step(state, reward)

	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

// PolicyExpectedSarsaLambda returns action based on tile coded state
func (agent *ESarsa) PolicyExpectedSarsaLambda(tileCodedStateActiveFeatures []int) (rlglue.Action, []float64) {
	probs := make([]float64, 2) // Probabilities of taking actions 0 and 1

	// Calculates action values
	actionValue0 := agent.ActionValue(tileCodedStateActiveFeatures, 0)
	actionValue1 := agent.ActionValue(tileCodedStateActiveFeatures, 1)

	// Calculates Epsilon-greedy probabilities for both actions
	if actionValue0 < actionValue1 {
		probs[0] = agent.epsilon / 2
		probs[1] = 1 - probs[0]

	} else {
		probs[0] = 1 - agent.epsilon/2
		probs[1] = 1 - probs[0]
	}

	// Random sampling action based on epsilon-greedy policy
	random := agent.rng.Float64()
	var action rlglue.Action
	if random < probs[0] {
		action = 0
	} else {
		action = 1
	}

	return action, probs
}

// ActionValue returns action value for a tile coded state and action pair
func (agent *ESarsa) ActionValue(tileCodedStateActiveFeatures []int, action rlglue.Action) float64 {
	var actionValue float64

	// Calculates action value as linear function (dot product) between weights and binary featured state
	for _, value := range tileCodedStateActiveFeatures {
		actionValue += agent.weights[action][value]
	}

	return actionValue
}
