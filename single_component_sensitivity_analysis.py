"""Purpose of this script is to examine the contribution of Diffusive motion, fission/fusion probabilities
percentage of biased fission events on DUMp organization and inheritance.
"""
import datetime
import pickle
from configs.kinetics_config import DELTA_T
from configs.structural_config import (
    PARTICLE_RADIUS, N_PARTICLES, AGGREGATE_PERCENT,
    MOTHER_BOX_SIDELENGTH, FUSION_EPSILON)

from lib.structures import Box
from lib.kinetics import (WT_DIFFUSION, DUMPN_DIFFUSION, DUMPP_DIFFUSION)
from lib.probability import (P_WT_EVENT, P_DUMPN_EVENT, P_DUMPP_EVENT)
from lib.simulation_handler import Simulation

SIMULATION_ENV = Box(lower_y=0, upper_y=MOTHER_BOX_SIDELENGTH,
                     lower_x=0, upper_x=MOTHER_BOX_SIDELENGTH)


def single_component_sensitivity_analysis(x, nAgg: int,
                                    da_trigger: bool,
                                    ffi_trigger: bool,
                                    targFis_probability: float,
                                    save_dir: str):
    """Function to test a variety of simulation parameters
    Will test fission/fusion rate changes, diffusive motion changes, targeted fission

    Args:
        x (int): simulation run number
        nAgg (int): number of discrete aggregates to seed simulation with
        da_trigger (bool): whether to change diffusive parameters into DUMP parameters at transition time
            Set to True to enable transition from WT to DUMP parameters for diffusive motion
        ffi_trigger (bool): whether to change fission/fusion parameters into DUMP parameters at transition time
            Set to True to enable transition from WT to DUMP parameters for fission/fusion probabilities 
        targFis_probability (float): Percentage of fission events that are targeted fission.  When set to 0,
            all fission is random, when set to 1, all fission is biased next to aggregate.  When set to 0.5,
            half of all fission events occur next to aggregate, half are random.
    """
    test = Simulation(particle_radius=PARTICLE_RADIUS,
                      n_particles=N_PARTICLES,
                      percent_agg=AGGREGATE_PERCENT,
                      min_interaction_dist=FUSION_EPSILON,
                      environment=SIMULATION_ENV,
                      diffusion_nonAgg_func=WT_DIFFUSION,
                      diffusion_Agg_func=WT_DIFFUSION,
                      p_event_nonAgg=P_WT_EVENT,
                      p_event_Agg=P_WT_EVENT,
                      targeted_fis_probability=targFis_probability,
                      indivisible_aggs=True,
                      dt=DELTA_T)

    # generate names
    if da_trigger:
        da_trigger_name = "dump"
    else:
        da_trigger_name = "wt"
    if ffi_trigger:
        ffi_trigger_name = "dump"
    else:
        ffi_trigger_name = "wt"
        
    
    if da_trigger_name == "dump" and ffi_trigger_name =="wt" and targFis_probability==0.0:
        pass
    elif da_trigger_name == "dump" and ffi_trigger_name =="dump":
        pass
    elif da_trigger_name == "wt" and ffi_trigger_name =="dump" and targFis_probability==0.0:
        pass
    else:
        return

    date_prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    # set all nodes to non-aggregate containing
    test.set_allNonagg()
    # Run for 10 min
    test.runUntilTimestamp(stopTime=1800)
    # Assign n discrete aggregate clusters
    test.set_allNonagg()
    test.assign_n_cluster_agg(nAgg)

    if da_trigger:
        test.diffusion_nonAgg_func = DUMPN_DIFFUSION
        test.diffusion_Agg_func = DUMPP_DIFFUSION
    if ffi_trigger:
        test.p_event_nonAgg = P_DUMPN_EVENT
        test.p_event_Agg = P_DUMPP_EVENT
        
        
    # Save simulation state prior to running simulation any further after transition
    with open(f'{save_dir}/{date_prefix}_clusterTestPBC2hr_nA{nAgg:02d}_{da_trigger_name}DA_{ffi_trigger_name}FFI_{int(targFis_probability*100):03d}targFis_s{x:03d}_PRE.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Run for another 20 min
    test.runUntilTimestamp(stopTime=7200)

    with open(f'{save_dir}/{date_prefix}_clusterTestPBC2hr_nA{nAgg:02d}_{da_trigger_name}DA_{ffi_trigger_name}FFI_{int(targFis_probability*100):03d}targFis_s{x:03d}_POST.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == "__main__":
    import sys
    import itertools
    from multiprocessing import Pool

    experiment_dir = [sys.argv[-1]]

    # Args for run for contrib of parameters =====================
    nSimulations = 500
    nClusters = [1,2,3,4,5,6]
    diffusion_trigger = [True, False]
    ffi_trigger = [True, False]
    targFis_probability_vals = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8,1.0]

    print("="*20, "\nRunning Simulations")
    print("="*20)
    args = itertools.product(list(range(nSimulations)),
                             nClusters,
                             diffusion_trigger,
                             ffi_trigger,
                             targFis_probability_vals,
                             experiment_dir)

    with Pool() as pool:
        results = pool.starmap(single_component_sensitivity_analysis, args)
