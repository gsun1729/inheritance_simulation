import numpy as np
import random

from lib.regression import LinearFunction, ExponentialFunction
from configs.structural_config import N_PARTICLES, PARTICLE_VOLUME

SIGMA = 0.1
class VolumeBasedMovement:
    # Function takes in number of particles
    def __init__(self, diff_const_equ: LinearFunction,
                 timestep: float) -> None:
        self.diffusion_const_equ = diff_const_equ
        self.dt = timestep

    def __call__(self, structure_size: int) -> np.ndarray:
        assert structure_size > 0, "Structure must contain at least one particle"
        assert structure_size <= N_PARTICLES, "Structure cannot contain more particles than in simulation"
        
        structure_volume = PARTICLE_VOLUME * structure_size

        D_coef = self.diffusion_const_equ(structure_volume)

        if D_coef < 0:
            return np.array([0, 0])

        L_d = np.sqrt(4*D_coef*(self.dt))
        L_d *= np.random.normal(loc=1, scale=SIGMA)

        random_angle = random.uniform(0, 2 * np.pi)
        
        xy_update = np.array(
            [np.cos(random_angle), np.sin(random_angle)]) * L_d

        return xy_update
    
    
class VolumeBasedFunctionMovement(ExponentialFunction):
    # Function takes in number of particles
    def __init__(self, m: float, t: float) -> None:
        super().__init__(m, t)
        
    def __call__(self, structure_size: int) -> float:
        magnitude = super().__call__(structure_size)
        
        
        random_scale = np.random.normal(loc = 1, scale=SIGMA)
        random_angle = random.uniform(0, 2 * np.pi)
        xy_update = np.array(
            [np.cos(random_angle), np.sin(random_angle)]) * magnitude * random_scale
        return xy_update


from configs.kinetics_config import f_wt_diffConstVsVol
from configs.kinetics_config import f_DUMPneg_diffConstVsVol
from configs.kinetics_config import f_DUMPpos_diffConstVsVol
from configs.kinetics_config import DELTA_T

WT_DIFFUSION = VolumeBasedMovement(diff_const_equ=f_wt_diffConstVsVol,
                                    timestep=DELTA_T)

DUMPN_DIFFUSION = VolumeBasedMovement(diff_const_equ=f_DUMPneg_diffConstVsVol,
                                        timestep=DELTA_T)

DUMPP_DIFFUSION = VolumeBasedMovement(diff_const_equ=f_DUMPpos_diffConstVsVol,
                                        timestep=DELTA_T)


