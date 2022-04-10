import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
import plot.loadFromEpisodeLengths as pel


def AUC(data):
    averageAcrossRuns = np.mean(data, axis=0)
    return np.sum(averageAcrossRuns)

def bottom50percentile(data):
    return np.sort(np.concatenate(data).ravel())[int(len(data)/2.0)]

def bottom10percentile(data):
    return np.sort(np.concatenate(data).ravel())[int(len(data)/2.0)]

def ranking(dirpath, run_number):
    subdirs = os.listdir(dirpath)
    performance = {}
    for s in range(len(subdirs)):
        print('---------> ' + str(((s+1)*100.0)/len(subdirs)))
        data = pel.load_data(dirpath+subdirs[s], run=run_number)
        convertedData, totalTimesteps = pel.convert_data('', data)

        transformation = 'Average-Rewards'
        window = 2500
        alpha = 0.0004
        averaging_type='exponential-averaging'

        transformedData = pel.transform_data('', convertedData, totalTimesteps, transformation, window, type=averaging_type, alpha=alpha)
        performance[subdirs[s]] = AUC(transformedData)
        # performance[subdirs[s]] = bottom10percentile(transformedData)
    print('-------------------------------------------------------')
    return (sorted(performance.items(), key=lambda item:item[1]))[::-1]


def plotPerformances(realvalues, modelvalues_lst, plt, rankingUnderModel_lst, rankingUnderReal, model_key_lst, title, compareVal=None, compareRank=None):
    cmap = matplotlib.cm.get_cmap('cool')

    scale=50000

    if compareVal is not None and compareRank is not None:
        comparevaluesRankedByRealRanking = [compareVal[compareRank.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
        plt.scatter([i+1 for i in range(len(compareVal))], np.array(comparevaluesRankedByRealRanking)/scale, color="black", label='Performance in\nthe vanilla model', s=3)
        print(comparevaluesRankedByRealRanking)
    # plt.scatter([i+1 for i in range(len(realvalues))], np.array(realvalues)/scale, label='real environment', s=5, color="blue")

    for j in range(len(modelvalues_lst)):
        key = model_key_lst[j]
        modelvaluesRankedByRealRanking = [modelvalues_lst[j][rankingUnderModel_lst[j].index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
        plt.scatter([i+1 for i in range(len(modelvalues_lst[j]))], np.array(modelvaluesRankedByRealRanking)/scale,
                    label='timeout={}'.format(key), s=5, color=cmap(j/len(modelvalues_lst)))
        print(modelvaluesRankedByRealRanking)

        max_idx = np.array(modelvaluesRankedByRealRanking).argmax()
        # plt.scatter([max_idx+1], np.array(modelvaluesRankedByRealRanking[max_idx]/scale), color="green", s=5)#, label='Hyperparameter\nchosen by model')
        plt.scatter([max_idx+1], np.array(modelvaluesRankedByRealRanking[max_idx]/scale), facecolors='none', edgecolors=cmap(j/len(modelvalues_lst)), s=160)

    #plt.scatter([i for i in range(len(modelvalues))], np.array(modelvalues)/50000, label='Average reward in the offline model')
    plt.xlabel('Hyperparameter ranking in the real environment', labelpad=35)
    plt.ylabel('Average reward\nof each\nhyperparameter setting\n(AUC)', rotation=0, labelpad=55)
    # plt.arrow(15, -0.04, -3, 0.035, color='black', width=0.00005, length_includes_head=True, head_length=0.002, head_width=0.002)
    # plt.text(5, -0.05, 'Hyperparameters chosen\nby the offline model', fontsize=8)
    plt.legend(loc=(0.01, 0.1), prop={'size': 8})
    #plt.tight_layout()
    plt.savefig('../img/auc_{}.png'.format(title),dpi=300, bbox_inches='tight')
    #plt.show()

def all_performance(model_parents_lst, real_parent, vanilla=None, title="default", run_number=0):
    realInfo = ranking(real_parent, run_number)

    realkeys = [key for (key, value) in realInfo]
    realvalues = [value for (key, value) in realInfo]
    rankingUnderReal = [i for i in range(len(realkeys))]

    if vanilla is not None:
        vanillaInfo = ranking(vanilla, run_number)
        vanillakeys = [key for (key, value) in vanillaInfo]
        vanillavalues = [value for (key, value) in vanillaInfo]
        rankingUnderVanilla = [realkeys.index(vanillakeys[i]) for i in range(len(vanillakeys))]
    else:
        vanillavalues, rankingUnderVanilla = None, None


    modelvalues_lst, rankingUnderModel_lst = [], []
    label_key_lst = []
    for model_parents in model_parents_lst:
        modelInfoDic = {}
        for idx, performance in enumerate(model_parents):
            one_modelInfo = ranking(performance, run_number)
            for (k, v) in one_modelInfo:
                if k not in modelInfoDic.keys():
                    modelInfoDic[k] = [v]
                else:
                    modelInfoDic[k].append(v)
        modelInfo = []
        for k in modelInfoDic:
            all = np.array(modelInfoDic[k])
            modelInfo.append((k, all.min()))

        modelkeys = [key for (key, value) in modelInfo]
        modelvalues = [value for (key, value) in modelInfo]
        rankingUnderModel = [realkeys.index(modelkeys[i]) for i in range(len(modelkeys))]

        correlation = np.corrcoef(rankingUnderReal, rankingUnderModel)
        print(idx, correlation[0][1])

        modelvalues_lst.append(modelvalues)
        rankingUnderModel_lst.append(rankingUnderModel)
        label_key_lst.append(model_parents[0].split("/timeout")[1].split("/")[0])
        # label_key_lst.append(model_parents[0].split("/noise")[1].split("/")[0])

    plt.figure()
    plotPerformances(realvalues, modelvalues_lst, plt, rankingUnderModel_lst, rankingUnderReal,
                     label_key_lst, title,
                     compareVal=vanillavalues, compareRank=rankingUnderVanilla)
    plt.close()
    plt.clf()



k1_drop1 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed5/",
]
k1_drop2 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed5/",
]
k1_drop3 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed5/",
]

k3_drop1 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed5/",
]
k3_drop2 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed5/",
]
k3_drop3 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed5/",
]

k5_drop1 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.1/ensembleseed5/",
]
k5_drop2 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.2/ensembleseed5/",
]
k5_drop3 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed1/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed2/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed3/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed4/",
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0.3/ensembleseed5/",
]

k1_drop0 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0/ensembleseed0/",
]
k3_drop0 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/lockat_baseline/drop0/ensembleseed0/",
]
k5_drop0 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/lockat_baseline/drop0/ensembleseed0/",
]

k1_time10 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ens-ep/k1/timeout10/esarsa/step7.5k_env/lockat_baseline/drop0/ensembleseed0/",
]
k1_time200 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ens-ep/k1/timeout200/esarsa/step7.5k_env/lockat_baseline/drop0/ensembleseed0/",
]
k1_time1000 = [
    "../../data/hyperparam/cartpole/offline_learning/knn-ens-ep/k1/timeout1000/esarsa/step7.5k_env/lockat_baseline/drop0/ensembleseed0/",
]

real_parent = "../../data/hyperparam/cartpole/online_learning/esarsa-adam/step50k/sweep/"
vanilla_parent = None

for run in range(30):
    all_performance([k1_time10, k1_time200, k1_time1000], real_parent, vanilla_parent, title="timeout_run{}".format(run), run_number=run)
