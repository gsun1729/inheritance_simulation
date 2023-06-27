from configs.kinetics_config import (f_wt_diffConstVsVol, 
                                     f_DUMPneg_diffConstVsVol,
                                     f_DUMPpos_diffConstVsVol)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from configs.structural_config import MOTHER_MITO_TOTAL_VOL
    size = np.linspace(0, MOTHER_MITO_TOTAL_VOL*2, 200)

    D_wt = np.array([f_wt_diffConstVsVol(
        s) if f_wt_diffConstVsVol(s) > 0 else 0 for s in size])
    D_dumpNeg = np.array([f_DUMPneg_diffConstVsVol(
        s) if f_DUMPneg_diffConstVsVol(s) > 0 else 0 for s in size])
    D_dumpPos = np.array([f_DUMPpos_diffConstVsVol(
        s) if f_DUMPpos_diffConstVsVol(s) > 0 else 0 for s in size])



    plt.plot(size, D_wt, color="k")
    plt.plot(size, D_dumpNeg, color="r")
    plt.plot(size, D_dumpPos, color="g")
    plt.axvline(MOTHER_MITO_TOTAL_VOL, color="b")
    plt.legend(['WT', 'DUMP-', 'DUMP+'])
    plt.title('D vs structure size')

    plt.show()
