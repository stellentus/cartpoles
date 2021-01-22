package environment

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

const (
	dt             = 0.2
	linkLength1    = 1.0
	linkLength2    = 1.0
	linkMass1      = 1.0
	linkMass2      = 1.0
	linkCOMPOS1    = 0.5
	linkCOMPOS2    = 0.5
	linkMOI        = 1.0
	maxVel1        = 4 * math.Pi
	maxVel2        = 9 * math.Pi
	torqueNoiseMax = 0.0
	stateMinV      = -0.1
	stateMaxV      = 0.1
)

type acrobotSettings struct {
	Seed         int64     `json:"seed"`
	Delays       []int     `json:"delays"`
	PercentNoise []float64 `json:"percent_noise"`
}

type Acrobot struct {
	logger.Debug
	acrobotSettings
	state              rlglue.State
	rng                *rand.Rand
	buffer             [][]float64
	bufferInsertIndex  []int
	availTorqueActions []float64
}

func init() {
	Add("acrobot", NewAcrobot)
}

func NewAcrobot(logger logger.Debug) (rlglue.Environment, error) {
	return &Acrobot{Debug: logger}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Acrobot) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.acrobotSettings)
	if err != nil {
		err = errors.New("environment.Acrobot settings error: " + err.Error())
		env.Message("err", err)
		return err
	}
	env.Seed += int64(run)
	env.rng = rand.New(rand.NewSource(env.Seed)) // Create a new rand source for reproducibility
	env.availTorqueActions = []float64{+1.0, 0.0, -1.0}

	if len(env.PercentNoise) == 1 {
		// Copy it for all dimensions
		noise := env.PercentNoise[0]
		env.PercentNoise = []float64{noise, noise, noise, noise, noise, noise}
	} else if len(env.PercentNoise) != 6 && len(env.PercentNoise) != 0 {
		err := fmt.Errorf("environment.Acrobot requires percent_noise to be length 6, 1, or 0, not length %d", len(env.PercentNoise))
		env.Message("err", err)
		return err
	}

	if len(env.Delays) == 1 {
		// Copy it for all dimensions
		delay := env.Delays[0]
		env.Delays = []int{delay, delay, delay, delay, delay, delay}
	} else if len(env.Delays) != 6 && len(env.Delays) != 0 {
		err := fmt.Errorf("environment.Acrobot requires delays to be length 6, 1, or 0, not length %d", len(env.Delays))
		env.Message("err", err)
		return err
	}

	// If noise is off, set array to nil
	totalNoise := 0.0
	for _, noise := range env.PercentNoise {
		totalNoise += noise
	}
	if totalNoise == 0.0 {
		env.PercentNoise = nil
	}

	// If delays are off, set array to nil
	totalDelay := 0
	for _, noise := range env.Delays {
		totalDelay += noise
	}
	if totalDelay == 0 {
		env.Delays = nil
	}

	env.state = make(rlglue.State, 4)

	if len(env.Delays) != 0 {
		env.buffer = make([][]float64, 6)
		for i := range env.buffer {
			env.buffer[i] = make([]float64, env.Delays[i])
		}
		env.bufferInsertIndex = make([]int, 6)
	}

	env.Message("acrobot settings", fmt.Sprintf("%+v", env.acrobotSettings))

	return nil
}

func (env *Acrobot) noisyState() rlglue.State {
	stateLowerBound := []float64{-1.0, -1.0, -1.0, -1.0, -maxVel1, -maxVel2}
	stateUpperBound := []float64{1.0, 1.0, 1.0, 1.0, maxVel1, maxVel2}

	state := []float64{math.Cos(env.state[0]), math.Sin(env.state[0]), math.Cos(env.state[1]), math.Sin(env.state[1]), env.state[2], env.state[3]}

	if len(env.PercentNoise) != 0 {
		// Only add noise if it's configured
		for i := range state {
			state[i] += env.randFloat(env.PercentNoise[i]*stateLowerBound[i], env.PercentNoise[i]*stateUpperBound[i])
			state[i] = env.clamp(state[i], stateLowerBound[i], stateUpperBound[i])
		}
	}
	return state
}

func (env *Acrobot) clamp(x float64, min float64, max float64) float64 {
	return math.Max(min, math.Min(x, max))
}

func (env *Acrobot) randomizeState() {
	for i := range env.state {
		env.state[i] = env.randFloat(stateMinV, stateMaxV)
	}
}

func (env *Acrobot) randFloat(min, max float64) float64 {
	return env.rng.Float64()*(max-min) + min
}

// Start returns an initial observation.
func (env *Acrobot) Start() rlglue.State {
	env.randomizeState()
	return env.getObservations()
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (env *Acrobot) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	s := make(rlglue.State, len(env.state))

	copy(s, env.state)
	action := act.(int)
	torque := env.availTorqueActions[action]
	sAugmented := append(s, torque)

	ns := env.rk4(sAugmented, [2]float64{0.0, dt})
	ns = ns[:4]

	ns[0] = env.wrap(ns[0], -math.Pi, math.Pi)
	ns[1] = env.wrap(ns[1], -math.Pi, math.Pi)
	ns[2] = env.bound(ns[2], -maxVel1, maxVel1)
	ns[3] = env.bound(ns[3], -maxVel2, maxVel2)

	copy(env.state, ns)
	done := (-math.Cos(env.state[0])-math.Cos(env.state[1]+env.state[0]) > 1.0)

	var reward float64
	if done {
		reward = 0.0
	} else {
		reward = -1.0
	}

	return env.getObservations(), reward, done
}

func (env *Acrobot) getObservations() rlglue.State {
	// Add noise to state to get observations
	observations := env.noisyState()

	// Add delays
	if len(env.Delays) != 0 {
		for i, obs := range observations {
			observations[i] = env.buffer[i][env.bufferInsertIndex[i]]                 // Load the delayed observation
			env.buffer[i][env.bufferInsertIndex[i]] = obs                             // Store the current state
			env.bufferInsertIndex[i] = (env.bufferInsertIndex[i] + 1) % env.Delays[i] // Update the insertion point
		}
	}

	return observations
}

// GetAttributes returns attributes for this environment.
func (env *Acrobot) GetAttributes() rlglue.Attributes {
	// Add elements to attributes.
	attributes := struct {
		NumAction  int       `json:"numberOfActions"`
		StateDim   int       `json:"stateDimension"`
		StateRange []float64 `json:"stateRange"`
	}{
		3,
		6,
		[]float64{2.0, 2.0, 2.0, 2.0, 2.0 * maxVel1, 2.0 * maxVel2},
	}

	attr, err := json.Marshal(&attributes)
	if err != nil {
		env.Message("err", "environment.Acrobot could not Marshal its JSON attributes: "+err.Error())
	}

	return attr
}

func (env *Acrobot) wrap(x float64, min float64, max float64) float64 {
	diff := max - min
	for x > max {
		x = x - diff
	}
	for x < min {
		x = x + diff
	}
	return x
}

func (env *Acrobot) bound(x float64, min float64, max float64) float64 {
	return math.Max(min, math.Min(x, max))
}

func (env *Acrobot) dsdt(saugmented rlglue.State) []float64 {
	m1 := linkMass1
	m2 := linkMass2
	l1 := linkLength1
	lc1 := linkCOMPOS1
	lc2 := linkCOMPOS2
	I1 := linkMOI
	I2 := linkMOI
	g := 9.8
	var a float64

	a = saugmented[len(saugmented)-1]
	s := saugmented[:len(saugmented)-1]

	theta1 := s[0]
	theta2 := s[1]
	dtheta1 := s[2]
	dtheta2 := s[3]

	d1 := m1*math.Pow(lc1, 2) + m2*(math.Pow(l1, 2)+math.Pow(lc2, 2)+2*l1*lc2*math.Cos(theta2)) + I1 + I2
	d2 := m2*(math.Pow(lc2, 2)+l1*lc2*math.Cos(theta2)) + I2
	phi2 := m2 * lc2 * g * math.Cos(theta1+theta2-math.Pi/2.0)
	phi1 := -m2*l1*lc2*math.Pow(dtheta2, 2)*math.Sin(theta2) - 2*m2*l1*lc2*dtheta2*dtheta1*math.Sin(theta2) + (m1*lc1+m2*l1)*g*math.Cos(theta1-math.Pi/2.0) + phi2

	ddtheta2 := (a + d2/d1*phi1 - m2*l1*lc2*math.Pow(dtheta1, 2)*math.Sin(theta2) - phi2) / (m2*math.Pow(lc2, 2) + I2 - math.Pow(d2, 2)/d1)
	ddtheta1 := -(d2*ddtheta2 + phi1) / d1
	return []float64{dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0}
}

func (env *Acrobot) rk4(y0 rlglue.State, t [2]float64) rlglue.State {
	var i int
	temp := make(rlglue.State, len(y0))
	y1 := make(rlglue.State, len(y0))

	dt := t[1] - t[0]
	dt2 := dt / 2.0
	k1 := env.dsdt(y0)

	copy(temp, y0)
	for i = range y0 {
		temp[i] = y0[i] + dt2*k1[i]
	}
	k2 := env.dsdt(temp)

	copy(temp, y0)
	for i = range y0 {
		temp[i] = y0[i] + dt2*k2[i]
	}
	k3 := env.dsdt(temp)

	copy(temp, y0)
	for i = range y0 {
		temp[i] = y0[i] + dt*k3[i]
	}
	k4 := env.dsdt(temp)

	copy(y1, y0)
	for i = range y0 {
		y1[i] = y0[i] + dt/6.0*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
	}

	return y1
}