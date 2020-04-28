package environment

import (
	"encoding/json"
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

const (
	ExampleActionMax       = 10
	ExampleStateMax        = ExampleActionMax * 10
	ExampleNumberOfActions = 2*ExampleActionMax + 1
)

// Example is just a dumb test environment.
// state is adjusted by the action. Valid values are integers [0, ExampleNumberOfActions).
// Reward is equal to the new state.
// When |state| is >= ExampleStateMax, it's reset to 0.
type Example struct {
	logger.Debug
	state int
}

func init() {
	Add("example", NewExample)
}

func NewExample(logger logger.Debug) (rlglue.Environment, error) {
	return &Example{Debug: logger}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Example) Initialize(attr rlglue.Attributes) error {
	var ss struct{ Seed int64 }
	err := json.Unmarshal(attr, &ss)
	if err != nil {
		env.Message("warning", "environment.Example seed wasn't available")
		ss.Seed = 0
	}
	rand.Seed(ss.Seed)

	env.state = rand.Intn(ExampleNumberOfActions) - ExampleActionMax

	env.Message("msg", "environment.Example Initialize", "seed", ss.Seed)

	return nil
}

// Start returns an initial observation.
func (env *Example) Start() rlglue.State {
	return env.stateSlice()
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (env *Example) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	action := int(act)
	if action < -ExampleActionMax || action > ExampleActionMax {
		return rlglue.State{}, 0, true //, error.New("example.Example action must be between -10 and 10")
	}
	env.state += action - 10
	if env.state >= ExampleStateMax || env.state <= -ExampleStateMax {
		env.state = 0
	}

	return env.stateSlice(), float64(env.state), false
}

// GetAttributes returns attributes for this environment.
func (env *Example) GetAttributes() rlglue.Attributes {
	return rlglue.Attributes(`{"numberOfActions":4}`)
	// TODO should be saved as attributes from a known struct
	// ExampleNumberOfActions:  ExampleNumberOfActions,
	// DimensionOfState: 1,
	// StateRange:       []rlglue.State{[]float64{float64(ExampleStateMax) * 2}},
}

func (env *Example) stateSlice() rlglue.State {
	return rlglue.State([]float64{float64(env.state)})
}
