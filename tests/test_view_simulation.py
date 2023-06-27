from lib.simulation_handler import Simulation
from configs.structural_config import DIFFERENCE_EPSILON, FUSION_EPSILON, FISSION_EPSILON
from lib.structures import Particle, Box
from copy import deepcopy
import pickle


if __name__ == "__main__":
    from configs.kinetics_config import DELTA_T
    from configs.structural_config import (
        PARTICLE_RADIUS, N_PARTICLES, AGGREGATE_PERCENT,
        MOTHER_BOX_SIDELENGTH, DIFFERENCE_EPSILON)

    from lib.kinetics import (WT_DIFFUSION, DUMPN_DIFFUSION, DUMPP_DIFFUSION)
    from lib.probability import (P_WT_EVENT, P_DUMPN_EVENT, P_DUMPP_EVENT)

    # from lib.kinetics import (DELDNM1_DIFFUSION, DELDNM1FZO1_DIFFUSION)

    SIMULATION_ENV = Box(lower_y=0, upper_y=MOTHER_BOX_SIDELENGTH,
                         lower_x=0, upper_x=MOTHER_BOX_SIDELENGTH)

    test = Simulation(particle_radius=PARTICLE_RADIUS,
                      n_particles=N_PARTICLES,
                      percent_agg=AGGREGATE_PERCENT,
                      min_interaction_dist=FUSION_EPSILON,
                      environment=SIMULATION_ENV,
                      diffusion_nonAgg_func=WT_DIFFUSION,
                      diffusion_Agg_func=WT_DIFFUSION,
                      p_event_nonAgg=P_WT_EVENT,
                      p_event_Agg=P_WT_EVENT,
                      targeted_fis_probability=0.5,
                      indivisible_aggs=False,
                      dt=DELTA_T)
    test.do_animation()