"""Purpose of this script is to run the simulation and test for mitochondria structure convergence using WT parameters
"""
import datetime
import pickle
from configs.kinetics_config import DELTA_T
from configs.structural_config import (
    PARTICLE_RADIUS, N_PARTICLES,
    MOTHER_BOX_SIDELENGTH, FUSION_EPSILON)
from lib.structures import Box
from lib.kinetics import WT_DIFFUSION
from lib.probability import P_WT_EVENT
from lib.simulation_handler import Simulation

SIMULATION_ENV = Box(lower_y=0, upper_y=MOTHER_BOX_SIDELENGTH,
                     lower_x=0, upper_x=MOTHER_BOX_SIDELENGTH)

def run4time_simulation(x: int, timestop: int, save_dir: str):
    """Function for running simulation for time
    Used to measure convergence of simulation discrete mitochondrial bodies over time

    Args:
        x (int): simulation run number
        timestop (int): timestamp to run until in seconds
        save_dir (str): directory to save simulation state results to
    """
    test = Simulation(particle_radius=PARTICLE_RADIUS,
                      n_particles=N_PARTICLES,
                      percent_agg=0,
                      min_interaction_dist=FUSION_EPSILON,
                      environment=SIMULATION_ENV,
                      diffusion_nonAgg_func=WT_DIFFUSION,
                      diffusion_Agg_func=WT_DIFFUSION,
                      p_event_nonAgg=P_WT_EVENT,
                      p_event_Agg=P_WT_EVENT,
                      targeted_fis_probability=0.0,
                      indivisible_aggs=False,
                      dt=DELTA_T)
    test.runUntilTimestamp(stopTime=timestop)
    date_prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    with open(f'{save_dir}/{date_prefix}_run4time_stopAt{timestop}_s{x:03d}.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == "__main__":
    import sys
    import itertools
    from multiprocessing import Pool

    experiment_dir = [sys.argv[-1]]

    # Args for run4time=================
    nSimulations = 200
    timestop = [10800]
    args = itertools.product(list(range(nSimulations)),
                             timestop,
                             experiment_dir)

    with Pool() as pool:
        results = pool.starmap(run4time_simulation, args)