import copy
import json
import numpy as np

def write_json(online_template, online_path, settings):
    config_dict = copy.deepcopy(online_template)
    for i, setting in enumerate(settings):
        for key, value in setting.items():
            config_dict["agent-settings"]["sweep"][key] = [value]
        config_dict["experiment-settings"]["data-path"] = online_template["experiment-settings"]["data-path"].format(i)
        new_conf_file_c = online_path.format(i)
        with open(new_conf_file_c, 'w') as conf:
            json.dump(config_dict, conf, indent=4)


def acrobot_500_subopt():
    online_template = {
        "agent-name": "esarsa",
        "environment-name": "acrobot",
        "agent-settings": {
            "gamma": 1.0,
            "state-len": 6,
            "env-name": "acrobot",
            "sweep": {
                "tilings": [16],
                "tiles": [8],
                "is-stepsize-adaptive": [True],
                "alpha": [0.0],
                "lambda": [0.8],
                "epsilon": [0.0],
                "adaptive-alpha": [],
                "beta1": [0.0],
                "softmax-temp": [],
                "weight-init": [0.0]
            },
            "lock-weight": False,
            "enable-debug": False,
            "seed": 1
        },
        "environment-settings": {
            "seed": 1
        },
        "experiment-settings": {
            "randomize_start_state_beforeLock": False,
            "randomize_start_state_afterLock": False,
            "steps": 0,
            "episodes": 1000000,
            "max-run-length-episodic": 15000,
            "data-path": "data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/randomsearch/param_{}",
            "should-log-totals": True,
            "debug-interval": 0
        }
    }
    # # 100 iterations
    settings = [
        {'adaptive-alpha':0.05865129348100832, 'softmax-temp':0.10063572018284903},
        {'adaptive-alpha':0.03976768369855336, 'softmax-temp':0.8268544501649522},
        {'adaptive-alpha':0.05442081601481022, 'softmax-temp':0.4105664018828965},
        {'adaptive-alpha':0.04404537176707395, 'softmax-temp':0.7844230056009689},
        {'adaptive-alpha':0.05956881170683345, 'softmax-temp':0.42339958894094926},
        {'adaptive-alpha':0.08248107204385255, 'softmax-temp':0.4711042406133956},
        {'adaptive-alpha':0.05667967713250513, 'softmax-temp':0.5646803821022534},
        {'adaptive-alpha':0.09312060196890218, 'softmax-temp':0.12459364782898504},
        {'adaptive-alpha':0.043046881829249053, 'softmax-temp':2.011816565050821},
        {'adaptive-alpha':0.04957732931341461, 'softmax-temp':0.669234261843487},
        {'adaptive-alpha':0.061573687603356936, 'softmax-temp':0.10592383934165772},
        {'adaptive-alpha':0.049749635274826245, 'softmax-temp':3.243646119203674},
        {'adaptive-alpha':0.03168079006890451, 'softmax-temp':1.2089770005467422},
        {'adaptive-alpha':0.024555910532465565, 'softmax-temp':0.7590181342561045},
        {'adaptive-alpha':0.028790485661731414, 'softmax-temp':2.309033177491888},
        {'adaptive-alpha':0.03713325657678248, 'softmax-temp':0.48874462519135065},
        {'adaptive-alpha':0.034149252806527086, 'softmax-temp':0.5887215271880524},
        {'adaptive-alpha':0.06439261219721384, 'softmax-temp':2.427046390610876},
        {'adaptive-alpha':0.025696842260383015, 'softmax-temp':0.1416267957699632},
        {'adaptive-alpha':0.012613087143994763, 'softmax-temp':0.49809575946792456},
        {'adaptive-alpha':0.031110332045303415, 'softmax-temp':1.3596421107779573},
        {'adaptive-alpha':0.08098333048674894, 'softmax-temp':1.6884468316600771},
        {'adaptive-alpha':0.07019378840943112, 'softmax-temp':1.487961576498566},
        {'adaptive-alpha':0.07986002073507065, 'softmax-temp':1.9211962910478784},
        {'adaptive-alpha':0.09018688780957156, 'softmax-temp':0.276213728224212},
        {'adaptive-alpha':0.05625867035394771, 'softmax-temp':2.0547157363338506},
        {'adaptive-alpha':0.027736457077925616, 'softmax-temp':0.48612360067525995},
        {'adaptive-alpha':0.04735435376740687, 'softmax-temp':0.6944650676270105},
        {'adaptive-alpha':0.044335180658624664, 'softmax-temp':0.12770858480531802},
    ]
    online_path = "config/hyperparam_v5/acrobot/online_learning/esarsa/step15k/randomsearch/param_{}.json"
    write_json(online_template, online_path, settings)

def puddlerand_500_subopt():
    online_template = {
        "agent-name": "esarsa",
        "environment-name": "puddleworld",
        "agent-settings": {
            "gamma": 1.0,
            "state-len": 2,
            "env-name": "puddleworld",
            "sweep": {
                "tilings": [16],
                "tiles": [8],
                "is-stepsize-adaptive": [True],
                "alpha": [0.0],
                "lambda": [0.1],
                "epsilon": [0.0],
                "adaptive-alpha": [],
                "beta1": [0.0],
                "softmax-temp": [],
                "weight-init": [0.0]
            },
            "lock-weight": False,
            "enable-debug": False,
            "seed": 1
        },
        "environment-settings": {
            "seed": 1
        },
        "experiment-settings": {
            "randomize_start_state_beforeLock": True,
            "randomize_start_state_afterLock": True,
            "steps": 0,
            "episodes": 1000000,
            "max-run-length-episodic": 30000,
            "data-path": "data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/randomsearch/param_{}",
            "should-log-totals": True,
            "should-log-returns": False,
            "debug-interval": 0
        }
    }
    # # 100 iterations
    settings = [
        {'adaptive-alpha':0.08289400292173632, 'softmax-temp':0.047950066449278114},
        {'adaptive-alpha':0.056103021925571, 'softmax-temp':0.1874542464400573},
        {'adaptive-alpha':0.07988855115046296, 'softmax-temp':0.3134443750903507},
        {'adaptive-alpha':0.08290010784747254, 'softmax-temp':0.7413039250036841},
        {'adaptive-alpha':0.09762744547762418, 'softmax-temp':0.06329632179069404},
        {'adaptive-alpha':0.09435400820158796, 'softmax-temp':1.1178125794085305},
        {'adaptive-alpha':0.09587226003159738, 'softmax-temp':0.6631385876535023},
        {'adaptive-alpha':0.09107134542882384, 'softmax-temp':0.3053622273895657},
        {'adaptive-alpha':0.07955745584289767, 'softmax-temp':0.3210495561015422},
        {'adaptive-alpha':0.09252000772711075, 'softmax-temp':0.39550872192499636},
        {'adaptive-alpha':0.05860219800531759, 'softmax-temp':0.37190703793084434},
        {'adaptive-alpha':0.0648046473172597, 'softmax-temp':0.5302755582982581},
        {'adaptive-alpha':0.09007148541170124, 'softmax-temp':0.3351808548358196},
        {'adaptive-alpha':0.09672531085605694, 'softmax-temp':1.4311177447986279},
        {'adaptive-alpha':0.08704276857248128, 'softmax-temp':0.08146143834964929},
        {'adaptive-alpha':0.09685052507178073, 'softmax-temp':0.791216021415945},
        {'adaptive-alpha':0.07800433424910276, 'softmax-temp':0.041839513447662104},
        {'adaptive-alpha':0.08293047639084283, 'softmax-temp':0.29254080820156875},
        {'adaptive-alpha':0.09884978760439249, 'softmax-temp':0.1111559318450453},
        {'adaptive-alpha':0.06671035088709226, 'softmax-temp':0.7017224554905492},
        {'adaptive-alpha':0.07752448939943628, 'softmax-temp':0.3676063999044365},
        {'adaptive-alpha':0.09708228029082584, 'softmax-temp':0.3563248468250271},
        {'adaptive-alpha':0.085845319585373, 'softmax-temp':0.05611860662400283},
        {'adaptive-alpha':0.08450938221263112, 'softmax-temp':0.6516893151328776},
        {'adaptive-alpha':0.07393777889149315, 'softmax-temp':0.09782981261689867},
        {'adaptive-alpha':0.099826756801199, 'softmax-temp':1.5009904941876917},
        {'adaptive-alpha':0.09686664884155977, 'softmax-temp':0.5105128929216207},
        {'adaptive-alpha':0.09970877855772964, 'softmax-temp':0.9441002323296555},
        {'adaptive-alpha':0.09713250996224297, 'softmax-temp':0.4898080505420607},
    ]
    online_path = "config/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/randomsearch/param_{}.json"
    write_json(online_template, online_path, settings)

if __name__ == '__main__':
    acrobot_500_subopt()
    puddlerand_500_subopt()