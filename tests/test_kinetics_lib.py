import numpy as np
from configs.structural_config import N_PARTICLES, PARTICLE_VOLUME
from lib.kinetics import WT_DIFFUSION, DUMPN_DIFFUSION, DUMPP_DIFFUSION

if __name__ == "__main__":
    from configs.structural_config import MOTHER_MITO_TOTAL_VOL, PARTICLE_VOLUME
    sizes = np.arange(1,N_PARTICLES)
    n_trials = 100
    out = []
    for s in sizes:
        wt = [np.linalg.norm(WT_DIFFUSION(s)) for i in range(n_trials)]
        dumpn = [np.linalg.norm(DUMPN_DIFFUSION(s)) for i in range(n_trials)]
        dumpp = [np.linalg.norm(DUMPP_DIFFUSION(s)) for i in range(n_trials)]
        wt = np.mean(wt)
        dumpn = np.mean(dumpn)
        dumpp = np.mean(dumpp)
        out.append([s, wt, dumpn, dumpp])
    out = np.array(out)

    import matplotlib.pyplot as plt

    plt.plot(out[:, 0] * PARTICLE_VOLUME, out[:, 1], color='k')
    plt.plot(out[:, 0] * PARTICLE_VOLUME, out[:, 2], color='r')
    plt.plot(out[:, 0] * PARTICLE_VOLUME, out[:, 3], color='g')
    plt.axvline(x=MOTHER_MITO_TOTAL_VOL)
    plt.legend(['WT', 'DUMP-', 'DUMP+'])
    plt.ylabel('Characteristic Step size ($\mu m$)')
    plt.xlabel('Volume ($\mu m^3$)')
    plt.title('Characteristic step size vs structure volume')
    plt.show()
