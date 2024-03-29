package agent

import (
	"encoding/json"
	"fmt"
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

type esarsaSettings struct {
	EnableDebug        bool         `json:"enable-debug"`
	Seed               int64        `json:"seed"`
	NumTilings         int          `json:"tilings"`
	NumTiles           int          `json:"tiles"`
	Gamma              float64      `json:"gamma"`
	Lambda             float64      `json:"lambda"`
	Epsilon            float64      `json:"epsilon"`
	Alpha              float64      `json:"alpha"`
	AdaptiveAlpha      float64      `json:"adaptive-alpha"`
	IsStepsizeAdaptive bool         `json:"is-stepsize-adaptive"`
	Scalers            [][2]float64 `json:"scalers"`
	NumActions         int          `json:"num-actions"`
}

// Expected sarsa-lambda with tile coding
type ESarsa struct {
	logger.Debug
	rng   *rand.Rand
	tiler util.MultiTiler

	// Agent accessible parameters
	weights                [][]float64 // weights is a slice of weights for each action
	traces                 [][]float64
	delta                  float64
	oldStateActiveFeatures []int
	oldAction              rlglue.Action
	stepsize               float64
	beta1                  float64
	beta2                  float64
	e                      float64
	m                      [][]float64
	v                      [][]float64
	timesteps              float64
	accumulatingbeta1      float64
	accumulatingbeta2      float64
	esarsaSettings
}

func init() {
	Add("esarsa", NewESarsa)
}

func NewESarsa(logger logger.Debug) (rlglue.Agent, error) {
	return &ESarsa{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *ESarsa) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	agent.esarsaSettings = esarsaSettings{
		// These default settings will be used if the config doesn't set these values
		NumTilings:         32,
		NumTiles:           4,
		Gamma:              0.99,
		Lambda:             0.8,
		Epsilon:            0.05,
		Alpha:              0.1,
		AdaptiveAlpha:      0.001,
		IsStepsizeAdaptive: false,
		Scalers: [][2]float64{
			{-maxPosition, maxPosition},
			{-maxVelocity, maxVelocity},
			{-maxAngle, maxAngle},
			{-maxAngularVelocity, maxAngularVelocity},
		},
		NumActions: 2,
	}

	err := json.Unmarshal(expAttr, &agent.esarsaSettings)
	if err != nil {
		agent.Message("warning", "agent.ESarsa settings weren't available: "+err.Error())
		agent.esarsaSettings.Seed = 0
	}

	if agent.IsStepsizeAdaptive == false {
		agent.stepsize = agent.Alpha / float64(agent.esarsaSettings.NumTilings) // Setting stepsize
	} else {
		agent.stepsize = agent.AdaptiveAlpha / float64(agent.esarsaSettings.NumTilings) // Setting adaptive stepsize
	}

	agent.beta1 = 0.9
	agent.beta2 = 0.999
	agent.e = math.Pow(10, -8)

	agent.esarsaSettings.Seed += int64(run)
	agent.rng = rand.New(rand.NewSource(agent.esarsaSettings.Seed)) // Create a new rand source for reproducibility

	// scales the input observations for tile-coding
	scalers := []util.Scaler{}
	for _, scale := range agent.esarsaSettings.Scalers {
		scalers = append(scalers, util.NewScaler(scale[0], scale[1], agent.esarsaSettings.NumTiles))
	}

	agent.tiler, err = util.NewMultiTiler(len(scalers), agent.esarsaSettings.NumTilings, scalers)
	if err != nil {
		return err
	}

	agent.weights = tilerArray(agent.esarsaSettings.NumActions, agent.tiler.NumberOfIndices()) // one weight slice for each action
	agent.traces = tilerArray(agent.esarsaSettings.NumActions, agent.tiler.NumberOfIndices())  // one trace slice for each action
	agent.m = tilerArray(agent.esarsaSettings.NumActions, agent.tiler.NumberOfIndices())
	agent.v = tilerArray(agent.esarsaSettings.NumActions, agent.tiler.NumberOfIndices())
	agent.timesteps = 0

	agent.Message("esarsa settings", fmt.Sprintf("%+v", agent.esarsaSettings))

	return nil
}

func tilerArray(actions, indices int) [][]float64 {
	a := make([][]float64, actions)
	for i := 0; i < actions; i++ {
		a[i] = make([]float64, indices)
	}
	return a
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *ESarsa) Start(state rlglue.State) rlglue.Action {
	var err error
	agent.oldStateActiveFeatures, err = agent.tiler.Tile(state) // Indices of active features of the tile-coded state

	if err != nil {
		agent.Message("err", "agent.ESarsa is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	agent.oldAction, _ = agent.PolicyExpectedSarsaLambda(agent.oldStateActiveFeatures) // Exp-Sarsa-L policy
	agent.timesteps++

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

	for j := range agent.weights {
		for _, value := range newStateActiveFeatures {
			agent.delta += agent.Gamma * epsilons[j] * agent.weights[j][value] // TD error target calculation
		}
	}

	var g float64
	var mhat float64
	var vhat float64

	// update for both actions for weights and traces
	for j := range agent.weights {
		for i := range agent.weights[j] {
			if agent.traces[j][i] != 0 { // update only where traces are non-zero
				if agent.IsStepsizeAdaptive == false {
					agent.weights[j][i] += agent.stepsize * agent.delta * agent.traces[j][i] // Semi-gradient descent, update weights
				} else {
					g = -agent.delta * agent.traces[j][i]
					agent.m[j][i] = agent.beta1*agent.m[j][i] + (1-agent.beta1)*g
					agent.v[j][i] = agent.beta1*agent.v[j][i] + (1-agent.beta1)*g*g

					mhat = agent.m[j][i] / (1 - math.Pow(agent.beta1, agent.timesteps))
					vhat = agent.v[j][i] / (1 - math.Pow(agent.beta2, agent.timesteps))
					agent.weights[j][i] -= agent.stepsize * mhat / (math.Pow(vhat, 0.5) + agent.e)
				}
				agent.traces[j][i] = agent.Gamma * agent.Lambda * agent.traces[j][i] // update traces
			}
		}
	}

	// New information is old for the next time step
	agent.oldStateActiveFeatures = newStateActiveFeatures
	agent.oldAction = newAction

	if agent.EnableDebug {
		agent.Message("msg", "step", "state", state, "reward", reward, "action", agent.oldAction)
	}

	agent.timesteps++
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
	greedyAction := 0
	greedyValue := agent.ActionValue(tileCodedStateActiveFeatures, 0)
	for i := 1; i < agent.NumActions; i++ {
		value := agent.ActionValue(tileCodedStateActiveFeatures, rlglue.Action(i))
		if value > greedyValue {
			greedyAction = i
			greedyValue = value
		}
	}

	// Calculates Epsilon-greedy probabilities for actions
	probs := make([]float64, agent.NumActions) // Probabilities of taking actions
	base := agent.Epsilon / float64(agent.NumActions)
	for i := range probs {
		probs[i] = base
	}
	probs[greedyAction] += 1 - agent.Epsilon

	var action int
	r := agent.rng.Float64() - probs[action]
	for r > 0 && action < agent.NumActions {
		action++
		r -= probs[action]
	}

	return rlglue.Action(action), probs
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
