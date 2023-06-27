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
    # corner corner test

    # test.particle_dict[0].position = np.array([0.31, 0.31])
    # test.particle_dict[1].position = np.array([8.3, 8.3])
    # test.particle_dict[2].position = np.array([8.3, 1.1])

    # test.particle_dict[0].position = np.array([8.3, 0.31])
    # test.particle_dict[1].position = np.array([0.31, 8.31])
    # edge test
    
    # test.particle_dict[0].position = np.array([8.3, 5])
    # test.particle_dict[1].position = np.array([7.7, 5.0])
    # test.particle_dict[2].position = np.array([8.3, 4.0])
    # center test



    # test.particle_dict[0].position = np.array([1.0, 1.0])
    # test.particle_dict[1].position = np.array([1.0, 2])
    # test.particle_dict[2].position = np.array([1, 2.7])
    # test.particle_dict[3].position = np.array([1.0, 7])
    # test.particle_dict[4].position = np.array([7, 1.5])
    test.set_allNonagg()
    test.drawSelf()
    test.runUntilTimestamp(100, print_progress=True)
    test.drawSelf()
    test.assign_n_cluster_agg(5)
    test.transition_time = deepcopy(test.timestamp)
    test.transition_snapshot = deepcopy(test.network)
    test.runUntilTimestamp(200, print_progress=True)
    # test.do_animation()
    # test.runNIter(5, print_progress=True)
    test.drawSelf()

    with open(f'test_simulation.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    raise

    test.runUntil(200, print_progress=True)
    test.set_allNonagg()
    test.drawSelf()
    pos = [p['pos'] for _, p in test.network.nodes(data=True)]
    c = [p['obj'].has_agg for _, p in test.network.nodes(data=True)]
    nx.draw(test.network, pos=pos, node_color=c, with_labels=True)
    plt.show()

    test.assign_n_cluster_agg(6)
    test.drawSelf()
    pos = [p['pos'] for _, p in test.network.nodes(data=True)]
    c = [p['obj'].has_agg for _, p in test.network.nodes(data=True)]
    nx.draw(test.network, pos=pos, node_color=c, with_labels=True)
    plt.show()

    test.runUntil(300, print_progress=True)
    test.drawSelf()
    pos = [p['pos'] for _, p in test.network.nodes(data=True)]
    c = [p['obj'].has_agg for _, p in test.network.nodes(data=True)]
    nx.draw(test.network, pos=pos, node_color=c, with_labels=True)
    plt.show()
    plt.plot(test.nAgg_history)
    plt.show()

    for k, v in test.network_history.items():
        pos = [p['pos'] for _, p in v.nodes(data=True)]
        c = [p['obj'].has_agg for _, p in v.nodes(data=True)]
        nx.draw(v, pos=pos, node_color=c, with_labels=True)
        plt.show()
    # test.drawReducedHistory()
    # for n, ndata in test.history.nodes(data=True):
    #     print(test.getPreOp(ndata["members"], ndata['timestamp']))
    # test.do_animation()
    raise
    # Test collisions
    # Test for fusion
    # test.particle_dict[0].position = np.array([2.9,7.5])
    # test.particle_dict[1].position = np.array([8.0,6])
    # test.particle_dict[2].position = np.array([3.0,3])
    # test.particle_dict[2].velocity = np.array([0,0.4])
    # test.particle_dict[0].velocity = np.array([0.3,0])
    # test.particle_dict[1].velocity = np.array([-0.3,0])

    # test for fission

    # test.particle_dict[0].position = np.array([3.0,7.0])
    # test.particle_dict[1].position = np.array([3.0,3.0])
    # test.particle_dict[2].position = np.array([7.0,7.0])

    # test.network.add_edge(0,1,sep=0)
    # test.network.add_edge(0,2,sep=0)

    # test.drawSelf()
    # test.advance()
    # test.drawSelf()
    # test.advance()
    # test.drawSelf()

    # test.runUntil(stopTime=600)

    # test.diffusion_nonAgg_func = DUMPN_DIFFUSION
    # test.diffusion_Agg_func = DUMPP_DIFFUSION
    # test.p_event_nonAgg = P_DUMPN_EVENT
    # test.p_event_Agg = P_DUMPP_EVENT
    # test.targeted_fis = False

    # test.runUntil(stopTime=1200)
    # test.drawReducedHistory()
    # test.drawSelf()

    # with open(f'simulation_runs/{date_prefix}_Simulation_dumpTransition_noTargFis.pickle', 'wb') as handle:
    #     pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
