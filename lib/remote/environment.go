package remote

import (
	"context"

	"github.com/stellentus/cartpoles/lib/rlglue"

	"google.golang.org/grpc"
)

type environmentServer struct {
	env rlglue.Environment
}

func NewEnvironmentServer(env rlglue.Environment) EnvironmentServer {
	return environmentServer{env}
}

func (srv environmentServer) Initialize(ctx context.Context, in *Attributes) (*Empty, error) {
	err := srv.env.Initialize(rlglue.Attributes(in.Attributes))
	return &Empty{}, err
}

func (srv environmentServer) Start(ctx context.Context, in *Empty) (*State, error) {
	state := srv.env.Start()
	return &State{Values: []float64(state)}, nil
}

func (srv environmentServer) Step(ctx context.Context, in *Action) (*StepResult, error) {
	state, reward, terminal := srv.env.Step(rlglue.Action(in.GetAction()))
	return &StepResult{State: &State{Values: []float64(state)}, Reward: reward, Terminal: terminal}, nil
}

func (srv environmentServer) GetAttributes(ctx context.Context, in *Empty) (*Attributes, error) {
	attr := srv.env.GetAttributes()
	return &Attributes{Attributes: string(attr)}, nil
}

type environmentWrapper struct {
	client EnvironmentClient
}

func NewEnvironment(cc *grpc.ClientConn) rlglue.Environment {
	return &environmentWrapper{NewEnvironmentClient(cc)}
}

func (env environmentWrapper) Initialize(attr rlglue.Attributes) error {
	ctx := context.Background()
	err := reattempt(func() error {
		_, err := env.client.Initialize(ctx, &Attributes{Attributes: string(attr)})
		return err
	})
	return err
}

// Start returns an initial observation.
func (env environmentWrapper) Start() rlglue.State {
	ctx := context.Background()
	state, _ := env.client.Start(ctx, &Empty{})
	return rlglue.State(state.Values)
}

// Step takes an action and provides the new observation, the resulting reward, and whether the state is terminal.
func (env environmentWrapper) Step(action rlglue.Action) (rlglue.State, float64, bool) {
	ctx := context.Background()
	result, _ := env.client.Step(ctx, &Action{Action: uint64(action)})
	return rlglue.State(result.State.Values), result.Reward, result.Terminal
}

// GetAttributes returns attributes for this environment.
func (env environmentWrapper) GetAttributes() rlglue.Attributes {
	ctx := context.Background()
	attr, _ := env.client.GetAttributes(ctx, &Empty{})
	return rlglue.Attributes(attr.Attributes)
}
