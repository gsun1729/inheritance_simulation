import numpy as np
from configs.structural_config import N_PARTICLES, PARTICLE_VOLUME
from lib.probability import ProbOperationVol, P_WT_EVENT, P_DUMPP_EVENT, P_DUMPN_EVENT

if __name__ == "__main__":

    # import matplotlib.pyplot as plt
    # x = np.arange(1,N_PARTICLES)
    # y_fis = np.array([P_DUMPP_EVENT(i)[0] for i in x])
    # y_fus = np.array([P_DUMPP_EVENT(i)[1] for i in x])
    # # y_idle = np.array([P_WT_EVENT(i)[-1] for i in x])

    # # y_fis = np.array([f_P_wt_fission(i) for i in x])
    # # raise
    # # y_fus = np.array([f_P_wt_fusion(i) for i in x])
    # # # y_idle = np.array([P_WT_EVENT(i)] for i in x])

    # plt.plot(x, y_fis, color = 'r')
    # plt.plot(x, y_fus, color = 'g')
    # # plt.plot(x, y_idle, color ='k')
    # plt.show()

    # raise
    import pandas as pd
    import matplotlib.pyplot as plt

    def testProbabilities(nParticles: np.ndarray, nIter: int,
                          pTable: ProbOperationVol,
                          savePath: str) -> pd.DataFrame:
        data = []
        for nP in nParticles:
            data_subframe = []
            for n in range(nIter):
                event = pTable(nP)
                if event == "fis":
                    data_subframe += [1]
                elif event == "fus":
                    data_subframe += [-1]
                else:
                    data_subframe += [0]
            data_subframe = [nP] + data_subframe
            data.append(data_subframe)
        percentages = []
        for row in data:
            n_idle = row[1:].count(0)
            n_fis = row[1:].count(1)
            n_fus = row[1:].count(-1)
            total = len(row)-1
            percentages.append(
                [row[0], n_idle/total, n_fus/total, n_fis/total])
        percentages = pd.DataFrame(percentages, columns=[
                                   "nP", "idle", "fus", "fis"])

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(
            9, 5), sharex=True, sharey=False)
        ax[0].plot(percentages['nP']*PARTICLE_VOLUME,
                   percentages['fis'], color='r')
        ax[0].plot(percentages['nP']*PARTICLE_VOLUME,
                   percentages['fus'], color='g')
        ax[1].plot(percentages['nP']*PARTICLE_VOLUME,
                   percentages['idle'], color='k')
        ax[0].set_xlabel('volume')
        ax[1].set_xlabel('volume')
        ax[0].set_ylabel('observed frequency')
        ax[1].set_ylabel('observed frequency')
        ax[0].legend(["fis", "fus"])
        ax[1].legend(["idle"])
        if savePath is not None:
            plt.savefig(savePath)
        else:
            plt.show()

    nParticles = np.arange(1, N_PARTICLES)
    nIter = 1000

    testProbabilities(nParticles, nIter, pTable=P_WT_EVENT,
                      savePath=None)
    testProbabilities(nParticles, nIter, pTable=P_DUMPN_EVENT,
                      savePath=None)
    testProbabilities(nParticles, nIter, pTable=P_DUMPP_EVENT,
                      savePath=None)
