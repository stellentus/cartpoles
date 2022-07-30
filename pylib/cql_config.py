import os
import json
import copy

def offline_generate(template, new):
    with open(template, "r") as f:
        template_cfg = json.load(f)
    for i in range(30):
        temp = copy.deepcopy(template_cfg)
        temp["fixed_parameters"]["exp_name"] = temp["fixed_parameters"]["exp_name"].format(i)
        temp["fixed_parameters"]["offline_data_path"]["data"] = temp["fixed_parameters"]["offline_data_path"]["data"].format(i)
        with open(new.format(i), "w") as fs:
            json.dump(temp, fs, indent=4)
    return

def online_generate(template, new):
    with open(template, "r") as f:
        template_cfg = json.load(f)
    for i in range(30):
        temp = copy.deepcopy(template_cfg)
        temp["fixed_parameters"]["exp_name"] = temp["fixed_parameters"]["exp_name"].format(i)
        vp = temp["fixed_parameters"]["val_fn_config"]["path"]
        parm = vp.split("sweep_{}")
        vp_new = parm[0]+"sweep_{}".format(i)+parm[1]
        temp["fixed_parameters"]["val_fn_config"]["path"] = vp_new
        temp["fixed_parameters"]["csv_path"] = temp["fixed_parameters"]["csv_path"].format(i)
        with open(new.format(i), "w") as fs:
            json.dump(temp, fs, indent=4)
    return

if __name__ == '__main__':
    # template = "experiment/config/test_v0/acrobot/cql_offline/data5k_eps0/sweep.json"
    # new = "experiment/config/test_v0/acrobot/cql_offline/data5k_eps0/sweep_{}.json"
    # offline_generate(template, new)

    template = "experiment/config/test_v0/acrobot/dqn/loadPi_dataEps0/cql_init_fix.json"
    new = "experiment/config/test_v0/acrobot/dqn/loadPi_dataEps0/cql_init_fix_{}.json"
    online_generate(template, new)

    template = "experiment/config/test_v0/acrobot_shift/dqn/loadPi_dataEps0/cql_init_fix.json"
    new = "experiment/config/test_v0/acrobot_shift/dqn/loadPi_dataEps0/cql_init_fix_{}.json"
    online_generate(template, new)

    # template = "experiment/config/test_v0/puddlerand/cql_offline/data5k_eps0/sweep.json"
    # new = "experiment/config/test_v0/puddlerand/cql_offline/data5k_eps0/sweep_{}.json"
    # offline_generate(template, new)

    template = "experiment/config/test_v0/puddlerand/dqn/loadPi_dataEps0/cql_init_fix.json"
    new = "experiment/config/test_v0/puddlerand/dqn/loadPi_dataEps0/cql_init_fix_{}.json"
    online_generate(template, new)
