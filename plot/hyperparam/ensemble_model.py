import os
import numpy as np
import matplotlib.pyplot as plt
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
import plot.loadFromEpisodeLengths as pel


def AUC(data):
    averageAcrossRuns = np.mean(data, axis=0)
    return np.sum(averageAcrossRuns)

def bottom50percentile(data):
    return np.sort(np.concatenate(data).ravel())[int(len(data)/2.0)]

def ranking(dirpath):
    subdirs = os.listdir(dirpath)
    performance = {}
    for s in range(len(subdirs)):
        print('---------> ' + str(((s+1)*100.0)/len(subdirs)))
        data = pel.load_data(dirpath+subdirs[s])
        convertedData, totalTimesteps = pel.convert_data('', data)

        transformation = 'Average-Rewards'
        window = 2500
        alpha = 0.0004
        averaging_type='exponential-averaging'

        transformedData = pel.transform_data('', convertedData, totalTimesteps, transformation, window, type=averaging_type, alpha=alpha)
        performance[subdirs[s]] = AUC(transformedData)
        #performance[subdirs[s]] = bottom50percentile(transformedData)
    print('-------------------------------------------------------')
    return (sorted(performance.items(), key=lambda item:item[1]))[::-1]


def plotPerformances(realvalues, modelvalues, plt, rankingUnderModel, rankingUnderReal, title, compareVal=None, compareRank=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    scale=50000

    plt.scatter([i+1 for i in range(len(realvalues))], np.array(realvalues)/scale, label='Performance in\nthe real environment', s=5, color="blue")
    if compareVal is not None and compareRank is not None:
        comparevaluesRankedByRealRanking = [compareVal[compareRank.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
        plt.scatter([i+1 for i in range(len(compareVal))], np.array(comparevaluesRankedByRealRanking)/scale, color="black", label='Performance in\nthe vanilla model', s=3)
        print(comparevaluesRankedByRealRanking)
    modelvaluesRankedByRealRanking = [modelvalues[rankingUnderModel.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelvalues))], np.array(modelvaluesRankedByRealRanking)/scale, label='Performance in\nthe ensemble model', s=5, color="orange")
    print(modelvaluesRankedByRealRanking)

    max_idx = np.array(modelvaluesRankedByRealRanking).argmax()
    # plt.scatter([max_idx+1], np.array(modelvaluesRankedByRealRanking[max_idx]/scale), color="green", label='Hyperparameter\nchosen by the model', s=5)
    plt.scatter([max_idx+1], np.array(modelvaluesRankedByRealRanking[max_idx]/scale), facecolors='none', edgecolors="orange", s=160)

    #plt.scatter([i for i in range(len(modelvalues))], np.array(modelvalues)/50000, label='Average reward in the offline model')
    plt.xlabel('Hyperparameter ranking in the real environment', labelpad=35)
    plt.ylabel('Average reward\nof each\nhyperparameter setting\n(AUC)', rotation=0, labelpad=55)
    # plt.arrow(15, -0.04, -3, 0.035, color='black', width=0.00005, length_includes_head=True, head_length=0.002, head_width=0.002)
    # plt.text(5, -0.05, 'Hyperparameters chosen\nby the offline model', fontsize=8)
    plt.legend(loc=(0.01, 0.1), prop={'size': 8})
    #plt.tight_layout()
    plt.savefig('../img/auc_{}.png'.format(title),dpi=300, bbox_inches='tight')
    #plt.show()

def all_performance(model_parents, real_parent, vanilla=None):
    realInfo = ranking(real_parent)

    realkeys = [key for (key, value) in realInfo]
    realvalues = [value for (key, value) in realInfo]

    if vanilla is not None:
        vanillaInfo = ranking(vanilla)
        vanillakeys = [key for (key, value) in vanillaInfo]
        vanillavalues = [value for (key, value) in vanillaInfo]
        rankingUnderVanilla = [realkeys.index(vanillakeys[i]) for i in range(len(vanillakeys))]
    else:
        vanillavalues, rankingUnderVanilla = None, None


    for idx, performance in enumerate(model_parents):
        modelInfoDic = {}
        one_modelInfo = ranking(performance)
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

    rankingUnderReal = [i for i in range(len(realkeys))]
    rankingUnderModel = [realkeys.index(modelkeys[i]) for i in range(len(modelkeys))]

    correlation = np.corrcoef(rankingUnderReal, rankingUnderModel)
    print(idx, correlation[0][1])

    plt.figure()
    plotPerformances(realvalues, modelvalues, plt, rankingUnderModel, rankingUnderReal, idx,
                     compareVal=vanillavalues, compareRank=rankingUnderVanilla)
    plt.close()
    plt.clf()

real_parent = "../../data/hyperparam/cartpole/online_learning/esarsa-adam/step50k/sweep/"
vanilla_parent = "../../data/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/lockat_baseline/drop0/ensembleseed0/"
all_performance([vanilla_parent], real_parent)