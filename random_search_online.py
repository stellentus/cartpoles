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
            "data-path": "data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/bayesopt/param_{}",
            "should-log-totals": True,
            "debug-interval": 0
        }
    }
    # # 100 iterations
    # settings = [
    #     {'adaptive-alpha': 0.025888403994508414, 'softmax-temp': 0.8930131079497657},
    #     {'adaptive-alpha': 0.0293949170863306, 'softmax-temp': 1.0689311032250093},
    #     {'adaptive-alpha': 0.03951541124694951, 'softmax-temp': 0.13288309310035037},
    #     {'adaptive-alpha': 0.01466216649027209, 'softmax-temp': 1.0439094840809517},
    #     {'adaptive-alpha': 0.0568364039763502, 'softmax-temp': 0.3350130620793751},
    #     {'adaptive-alpha': 0.04659678817246602, 'softmax-temp': 0.8904879070262148},
    #     {'adaptive-alpha': 0.0786624229840085, 'softmax-temp': 0.12032497614049892},
    #     {'adaptive-alpha': 0.09620693212540682, 'softmax-temp': 0.22037629939666048},
    #     {'adaptive-alpha': 0.043046881829249053, 'softmax-temp': 2.011816565050821},
    #     {'adaptive-alpha': 0.04957732931341461, 'softmax-temp': 0.669234261843487},
    #     {'adaptive-alpha': 0.019380031517111086, 'softmax-temp': 0.45442759348461426},
    #     {'adaptive-alpha': 0.040262247229026875, 'softmax-temp': 1.1972217966364709},
    #     {'adaptive-alpha': 0.048273582910803636, 'softmax-temp': 0.1852709369632509},
    #     {'adaptive-alpha': 0.030715669309541227, 'softmax-temp': 1.229098212469371},
    #     {'adaptive-alpha': 0.06476411580413506, 'softmax-temp': 1.114566608325627},
    #     {'adaptive-alpha': 0.07798105912770517, 'softmax-temp': 1.0850961872770017},
    #     {'adaptive-alpha': 0.048172711864688966, 'softmax-temp': 2.04726641747713},
    #     {'adaptive-alpha': 0.051818631511344186, 'softmax-temp': 0.023141519237423865},
    #     {'adaptive-alpha': 0.0264387655858784, 'softmax-temp': 0.14161702404206614},
    #     {'adaptive-alpha': 0.00929629411834526, 'softmax-temp': 0.24207393423811685},
    #     {'adaptive-alpha': 0.030709419549241758, 'softmax-temp': 0.5751722405688467},
    #     {'adaptive-alpha': 0.07652954704460171, 'softmax-temp': 1.9993807218852366},
    #     {'adaptive-alpha': 0.09512779359183714, 'softmax-temp': 3.6468219027778073},
    #     {'adaptive-alpha': 0.03997567065983027, 'softmax-temp': 1.9217941529921618},
    #     {'adaptive-alpha': 0.09952769397150724, 'softmax-temp': 1.1011175746790913},
    #     {'adaptive-alpha': 0.041778078593283134, 'softmax-temp': 0.5861452846120763},
    #     {'adaptive-alpha': 0.03493675559172365, 'softmax-temp': 1.552082675155012},
    #     {'adaptive-alpha': 0.08168686796981035, 'softmax-temp': 0.26662109492929453},
    #     {'adaptive-alpha': 0.08789190939624143, 'softmax-temp': 0.011148122993051507},
    #     {'adaptive-alpha': 0.09008076595492581, 'softmax-temp': 0.801453463893327},
    # ]
    settings = [
        {'adaptive-alpha': 0.03943547210987053, 'softmax-temp': 1.6916760420700883},
        {'adaptive-alpha': 0.01352802308589306, 'softmax-temp': 0.46085696574666707},
        {'adaptive-alpha': 0.03951541124694951, 'softmax-temp': 0.13288309310035037},
        {'adaptive-alpha': 0.03224087365772787, 'softmax-temp': 1.0123233738378943},
        {'adaptive-alpha': 0.04139871683994211, 'softmax-temp': 0.8166895112778376},
        {'adaptive-alpha': 0.04031498381654247, 'softmax-temp': 0.8960613396317665},
        {'adaptive-alpha': 0.07844294750836714, 'softmax-temp': 0.11694638582942563},
        {'adaptive-alpha': 0.07460844223128153, 'softmax-temp': 0.1863751023695669},
        {'adaptive-alpha': 0.043046881829249053, 'softmax-temp': 2.011816565050821},
        {'adaptive-alpha': 0.04957732931341461, 'softmax-temp': 0.669234261843487},
        {'adaptive-alpha': 0.017356751604084477, 'softmax-temp': 0.4410694371454321},
        {'adaptive-alpha': 0.040262247229026875, 'softmax-temp': 1.1972217966364709},
        {'adaptive-alpha': 0.048273582910803636, 'softmax-temp': 0.1852709369632509},
        {'adaptive-alpha': 0.01676320083097572, 'softmax-temp': 0.6329503119884122},
        {'adaptive-alpha': 0.06476411580413506, 'softmax-temp': 1.114566608325627},
        {'adaptive-alpha': 0.0320067345326193, 'softmax-temp': 1.1146629392773801},
        {'adaptive-alpha': 0.048172711864688966, 'softmax-temp': 2.04726641747713},
        {'adaptive-alpha': 0.051818631511344186, 'softmax-temp': 0.023141519237423865},
        {'adaptive-alpha': 0.038057508435692305, 'softmax-temp': 0.16636135631731866},
        {'adaptive-alpha': 0.00929629411834526, 'softmax-temp': 0.24207393423811685},
        {'adaptive-alpha': 0.030709419549241758, 'softmax-temp': 0.5751722405688467},
        {'adaptive-alpha': 0.04207352406249908, 'softmax-temp': 2.249413413754938},
        {'adaptive-alpha': 0.08316653634571691, 'softmax-temp': 0.8860964913376786},
        {'adaptive-alpha': 0.06712091641518526, 'softmax-temp': 1.8457503837030595},
        {'adaptive-alpha': 0.09947981012875014, 'softmax-temp': 1.1010723484864335},
        {'adaptive-alpha': 0.041778078593283134, 'softmax-temp': 0.5861452846120763},
        {'adaptive-alpha': 0.02742973799439006, 'softmax-temp': 0.7935654836963727},
        {'adaptive-alpha': 0.08168686796981035, 'softmax-temp': 0.26662109492929453},
        {'adaptive-alpha': 0.08879156330629841, 'softmax-temp': 0.013350264336797188},
        {'adaptive-alpha': 0.0901037672375908, 'softmax-temp': 0.8029330545365461},
    ]
    online_path = "config/hyperparam_v5/acrobot/online_learning/esarsa/step15k/bayesopt/param_{}.json"
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
            "data-path": "data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/bayesopt/param_{}",
            "should-log-totals": True,
            "should-log-returns": False,
            "debug-interval": 0
        }
    }
    # # 100 iterations
    # settings = [
    #     {'adaptive-alpha': 0.0999964140464299, 'softmax-temp': 1.327731978457611},
    #     {'adaptive-alpha': 0.09529823452239909, 'softmax-temp': 0.590493037961509},
    #     {'adaptive-alpha': 0.09830009577379295, 'softmax-temp': 0.1500804154007128},
    #     {'adaptive-alpha': 0.0937859628446671, 'softmax-temp': 0.15819501314263176},
    #     {'adaptive-alpha': 0.09835286434053779, 'softmax-temp': 0.05577183563554051},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.14122862954357965},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
    #     {'adaptive-alpha': 0.09910939472768215, 'softmax-temp': 0.307933280145168},
    #     {'adaptive-alpha': 0.09984715809385092, 'softmax-temp': 0.5686857741581581},
    #     {'adaptive-alpha': 0.09178187551657728, 'softmax-temp': 0.15501143326064984},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.3352936036462956},
    #     {'adaptive-alpha': 0.09615371412851276, 'softmax-temp': 0.19393494133521402},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.12505078074515233},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.6782870216898292},
    #     {'adaptive-alpha': 0.09597995163527717, 'softmax-temp': 0.07202802109144721},
    #     {'adaptive-alpha': 0.09929035084298099, 'softmax-temp': 0.0722848785034774},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.6365744360776031},
    #     {'adaptive-alpha': 0.09951664950559322, 'softmax-temp': 0.18695916379046695},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.2953788964535545},
    #     {'adaptive-alpha': 0.09741431429619969, 'softmax-temp': 0.7562794205241162},
    #     {'adaptive-alpha': 0.0968250103259614, 'softmax-temp': 1.0040859011030858},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
    #     {'adaptive-alpha': 0.09499729107136673, 'softmax-temp': 0.719059685308171},
    #     {'adaptive-alpha': 0.09682806436064773, 'softmax-temp': 0.630584117742087},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.6455191738157036},
    #     {'adaptive-alpha': 0.09661427113685322, 'softmax-temp': 1.186352803325197},
    #     {'adaptive-alpha': 0.09951979048830613, 'softmax-temp': 0.5005392243575281},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.08521364530163601},
    #     {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
    #     {'adaptive-alpha': 0.09925710560788817, 'softmax-temp': 0.4266034591356373},
    # ]
    settings = [
        {'adaptive-alpha': 0.09710710723476351, 'softmax-temp': 0.24521072103072286},
        {'adaptive-alpha': 0.09529823452239909, 'softmax-temp': 0.590493037961509},
        {'adaptive-alpha': 0.09830009577379295, 'softmax-temp': 0.1500804154007128},
        {'adaptive-alpha': 0.0937859628446671, 'softmax-temp': 0.15819501314263176},
        {'adaptive-alpha': 0.09835286434053779, 'softmax-temp': 0.05577183563554051},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.14122862954357965},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
        {'adaptive-alpha': 0.09869889449083107, 'softmax-temp': 0.024239507091946225},
        {'adaptive-alpha': 0.09104290081248477, 'softmax-temp': 0.12704557391306362},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.14238392430911728},
        {'adaptive-alpha': 0.09444926775446741, 'softmax-temp': 0.24980633752807585},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.15013172927061552},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.5300920334516382},
        {'adaptive-alpha': 0.09597995163527717, 'softmax-temp': 0.07202802109144721},
        {'adaptive-alpha': 0.09410416581869012, 'softmax-temp': 0.32676838466383507},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.6368384216890978},
        {'adaptive-alpha': 0.094493645269151, 'softmax-temp': 0.13505170676622724},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.29113254342335043},
        {'adaptive-alpha': 0.099808065546431, 'softmax-temp': 0.43767068019201716},
        {'adaptive-alpha': 0.09978917056768793, 'softmax-temp': 0.5050771753738336},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.4806976367034401},
        {'adaptive-alpha': 0.09682806436064773, 'softmax-temp': 0.630584117742087},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.09056445562561083},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.06617122866465149},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.41881005435811014},
        {'adaptive-alpha': 0.1, 'softmax-temp': 0.001},
        {'adaptive-alpha': 0.09638800323251892, 'softmax-temp': 0.21045885044013612},
    ]
    online_path = "config/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/bayesopt/param_{}.json"
    write_json(online_template, online_path, settings)

if __name__ == '__main__':
    acrobot_500_subopt()
    puddlerand_500_subopt()