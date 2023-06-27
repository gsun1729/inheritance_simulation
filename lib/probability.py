from configs.probability_config import (
    f_P_dumpPos_fission, f_P_dumpPos_fusion)
from configs.probability_config import (
    f_P_dumpNeg_fission, f_P_dumpNeg_fusion)
from configs.probability_config import (f_P_wt_fission, f_P_wt_fusion)
import numpy as np
from lib.regression import DataBasedFunction
from configs.structural_config import N_PARTICLES, PARTICLE_VOLUME


def randomUniformBool(threshold):
    """Given a cutoff threshold, returns True if randomly generated value is larger than that of
    threshold, False otherwise
    Parameters
    ==========
        threshold (float) : cutoff threshold
    Returns
    ==========
        (bool)
    """
    prediction = np.random.uniform(0, 1)
    if prediction >= threshold:
        return False
    return prediction


def randomUniformTriOption(fission_interval_size: float, fusion_interval_size: float) -> str:
    """Given a fission and fusion probability, return whether fission, fusion, or nothing
    happens. 
    Fission interval : [0, fission_interval_size]
    None interval : (fission_interval_size, fusion_interval_size)
    and Fusion interval : [fusion_interval_size,1)

    Args:
        fission_interval_size (float): fission probability per timestep
        fusion_interval_size (float): fusion probability per timestep

    Returns:
        str: operation ("fis", "fus", "idle")
    """
    prediction = np.random.uniform(0, 1)
    if prediction <= fission_interval_size:
        return "fis"
    elif prediction >= 1 - fusion_interval_size:
        return "fus"
    else:
        return "idle"


class ProbOperationVol:
    # Function takes in number of particles; and outputs event
    def __init__(self, prob_fission_equ: DataBasedFunction,
                 prob_fusion_equ: DataBasedFunction) -> None:
        self.prob_fission_equ = prob_fission_equ
        self.prob_fusion_equ = prob_fusion_equ

    def __call__(self, structure_size: int) -> str:

        assert structure_size > 0, "Cannot have negative or zero length structure"
        assert structure_size <= N_PARTICLES, "Cannot have more than N_PARTICLES in simulation"

        # Get structure volume from particle volume;
        structure_volume = PARTICLE_VOLUME * structure_size

        p_fis = self.prob_fission_equ(structure_volume)
        p_fus = self.prob_fusion_equ(structure_volume)

        if structure_size == 1:
            p = np.random.choice(a=["fus", "idle"],
                                 p=[p_fus, 1-p_fus])
        elif structure_size == N_PARTICLES:
            p = np.random.choice(a=["fis", "idle"],
                                 p=[p_fis, 1-p_fis])
        else:
            p = np.random.choice(a=["fis", "fus", "idle"],
                                 p=[p_fis, p_fus, 1-p_fis-p_fus])
        # return [p_fis, p_fus, 1-p_fus-p_fis]
        return p


P_WT_EVENT = ProbOperationVol(prob_fission_equ=f_P_wt_fission,
                              prob_fusion_equ=f_P_wt_fusion)

P_DUMPN_EVENT = ProbOperationVol(prob_fission_equ=f_P_dumpNeg_fission,
                                 prob_fusion_equ=f_P_dumpNeg_fusion)

P_DUMPP_EVENT = ProbOperationVol(prob_fission_equ=f_P_dumpPos_fission,
                                 prob_fusion_equ=f_P_dumpPos_fusion)

