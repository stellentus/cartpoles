package state

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// SensorDriftWrapper is used to wrap an environment, adding sensor drift.
type SensorDriftWrapper struct {
	logger.Debug
	Env  rlglue.Environment
	Seed int64 `json:"seed"`
	// WrappedEnvAttrs is information loaded from the wrapped environment.
	// The wrapped environment must provide all of the JSON attributes in WrappedEnvAttrs, otherwise Initialize will return an error.
	WrappedEnvAttrs struct {
		// NumAction is the number of actions that can be taken in the environment.
		NumAction int64 `json:"numberOfActions"`
		// StateDim is the dimension of the observation array.
		StateDim int `json:"stateDimension"`
		// StateRange is the range of observations, obtained by (max observation - min observation).
		StateRange []float64 `json:"stateRange"`
	}
	// DriftAttrs is information loaded from the json config file.
	DriftAttrs struct {
		// DriftScale is the scale of the drift with respect to the state range.
		DriftScale float64 `json:"driftScale"`
		// SensorLife is the number of time-steps after which the probability of drift will become nearly 1
		SensorLife []float64 `json:"sensorLife"`
		// DriftProb is the probability scale, if it's less than 0, drift occurs with prob=1, if it's between 0 and 1, the max prob is scaled by this value.
		DriftProb []float64 `json:"driftProb"`
	}
	// noise is the zero-mean Gaussian noise applied to the observation at each time-step.
	noise []float64
	// noiseStd is the standard deviation of the noise generated at each time-step.
	noiseStd []float64
	// noiseMax is the maximum noise value used for clamping the noise and observation.
	noiseMax []float64
	// sensorSteps is the number of sensor drift time-steps.
	sensorSteps int64
	noiseFns    []func(int) float64
	rng         *rand.Rand
}

func init() {
	Add("sensor-drift", NewSensorDriftWrapper)
}

func NewSensorDriftWrapper(logger logger.Debug, env rlglue.Environment) (rlglue.Environment, error) {
	return &SensorDriftWrapper{Debug: logger, Env: env}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (wrapper *SensorDriftWrapper) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &wrapper)
	if err != nil {
		err = errors.New("environment.SensorDriftWrapper settings error: " + err.Error())
		wrapper.Message("err", err)
		return err
	}
	wrapper.rng = rand.New(rand.NewSource(wrapper.Seed + int64(run)))
	err = json.Unmarshal(attr, &wrapper.DriftAttrs)
	if err != nil {
		err = errors.New("environment.SensorDriftWrapper settings error: " + err.Error())
		wrapper.Message("err", err)
		return err
	}
	if wrapper.DriftAttrs.DriftScale == 0 || wrapper.DriftAttrs.SensorLife == nil || wrapper.DriftAttrs.DriftProb == nil {
		err = errors.New("environment.SensorDriftWrapper settings error: Invalid sensor drift attribute(s) in the config file")
		wrapper.Message("err", err)
		return err
	}
	envAttr := wrapper.Env.GetAttributes()
	err = json.Unmarshal(envAttr, &wrapper.WrappedEnvAttrs)
	if err != nil {
		err = errors.New("environment.SensorDriftWrapper settings error: " + err.Error())
		wrapper.Message("err", err)
		return err
	}
	if wrapper.WrappedEnvAttrs.NumAction == 0 || wrapper.WrappedEnvAttrs.StateDim == 0 || wrapper.WrappedEnvAttrs.StateRange == nil {
		err = errors.New("environment.SensorDriftWrapper settings error: Invalid attribute(s) from the wrapped environment")
		wrapper.Message("err", err)
		return err
	}

	wrapper.noise = make([]float64, wrapper.WrappedEnvAttrs.StateDim)
	wrapper.noiseStd = make([]float64, wrapper.WrappedEnvAttrs.StateDim)
	for i := range wrapper.noiseStd {
		wrapper.noiseStd[i] = wrapper.WrappedEnvAttrs.StateRange[i] / wrapper.DriftAttrs.DriftScale
	}
	wrapper.noiseMax = make([]float64, wrapper.WrappedEnvAttrs.StateDim)
	for i := range wrapper.noiseMax {
		wrapper.noiseMax[i] = wrapper.WrappedEnvAttrs.StateRange[i] / 2
	}
	wrapper.noiseFns = make([]func(int) float64, wrapper.WrappedEnvAttrs.StateDim)
	for i := range wrapper.DriftAttrs.DriftProb {
		if wrapper.DriftAttrs.DriftProb[i] < 0 {
			wrapper.noiseFns[i] = wrapper.gaussNoise
		} else {
			wrapper.noiseFns[i] = wrapper.probGaussNoise
		}
	}

	wrapper.Message("msg", "environment.SensorDriftWrapper Initialize",
		"drift scale", wrapper.DriftAttrs.DriftScale,
		"sensor life", wrapper.DriftAttrs.SensorLife,
		"drift probability", wrapper.DriftAttrs.DriftProb)
	return nil
}

// Start returns an initial observation.
func (wrapper *SensorDriftWrapper) Start() rlglue.State {
	return wrapper.Env.Start()
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (wrapper *SensorDriftWrapper) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	state, reward, done := wrapper.Env.Step(act)
	state = wrapper.stateProcess(state)
	return state, reward, done
}

// GetAttributes returns attributes for this environment.
func (wrapper *SensorDriftWrapper) GetAttributes() rlglue.Attributes {
	return wrapper.Env.GetAttributes()
}

func (wrapper *SensorDriftWrapper) stateProcess(state rlglue.State) rlglue.State {
	wrapper.sensorSteps++
	for i := 0; i < wrapper.WrappedEnvAttrs.StateDim; i++ {
		wrapper.noise[i] = wrapper.clamp(wrapper.noise[i]+wrapper.noiseFns[i](i), i)
		state[i] = wrapper.clamp(state[i]+wrapper.noise[i], i)
	}
	return state
}

func (wrapper *SensorDriftWrapper) gaussNoise(idx int) float64 {
	return wrapper.rng.NormFloat64() * wrapper.noiseStd[idx]
}

func (wrapper *SensorDriftWrapper) probGaussNoise(idx int) float64 {
	var noise float64
	probDrift := wrapper.logisticProb(idx)
	if sample := wrapper.rng.Float64(); sample < probDrift {
		noise = wrapper.rng.NormFloat64() * wrapper.noiseStd[idx]
	}
	return noise
}

func (wrapper *SensorDriftWrapper) logisticProb(idx int) float64 {
	return wrapper.DriftAttrs.DriftProb[idx] / (1 + math.Exp(-(float64(wrapper.sensorSteps)-wrapper.DriftAttrs.SensorLife[idx]/2)/(wrapper.DriftAttrs.SensorLife[idx]/10)))
}

func (wrapper *SensorDriftWrapper) clamp(x float64, idx int) float64 {
	return math.Max(-wrapper.noiseMax[idx], math.Min(x, wrapper.noiseMax[idx]))
}