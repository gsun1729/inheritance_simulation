from configs.probability_config import (f_P_wt_fission,
                                        f_P_wt_fusion,
                                        f_P_dumpNeg_fission,
                                        f_P_dumpNeg_fusion,
                                        f_P_dumpPos_fission,
                                        f_P_dumpPos_fusion)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from configs.structural_config import MOTHER_MITO_TOTAL_VOL
    x = np.linspace(0, MOTHER_MITO_TOTAL_VOL*2, 100)
    print(max(x))
    p_wt_fis = np.array([f_P_wt_fission(i) for i in x])
    p_wt_fus = np.array([f_P_wt_fusion(i) for i in x])

    p_dumpn_fis = np.array([f_P_dumpNeg_fission(i) for i in x])
    p_dumpn_fus = np.array([f_P_dumpNeg_fusion(i) for i in x])

    p_dumpp_fis = np.array([f_P_dumpPos_fission(i) for i in x])
    p_dumpp_fus = np.array([f_P_dumpPos_fusion(i) for i in x])

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(
        9, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].plot(x, p_wt_fis, color='r')
    ax[0].plot(x, p_wt_fus, color='g')
    ax[0].set_title('WT')
    ax[0].set_xlabel('volume ($\mu m^3$)')
    ax[0].set_ylabel('P per 0.1s')
    ax[0].legend(['fis', 'fus'])
    ax[0].axvline(x=MOTHER_MITO_TOTAL_VOL)
    ax[0].set_xlim(left=0)

    ax[1].plot(x, p_dumpn_fis, color='r')
    ax[1].plot(x, p_dumpn_fus, color='g')
    ax[1].set_title('DUMP-')
    ax[1].set_xlabel('volume ($\mu m^3$)')
    ax[1].legend(['fis', 'fus'])
    ax[1].axvline(x=MOTHER_MITO_TOTAL_VOL)
    ax[1].set_xlim(left=0)

    ax[2].plot(x, p_dumpp_fis, color='r')
    ax[2].plot(x, p_dumpp_fus, color='g')
    ax[2].set_title('DUMP+')
    ax[2].set_xlabel('volume ($\mu m^3$)')
    ax[2].legend(['fis', 'fus'])
    ax[2].axvline(x=MOTHER_MITO_TOTAL_VOL)
    ax[2].set_xlim(left=0)

    plt.show()
