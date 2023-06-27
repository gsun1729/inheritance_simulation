"""Test percent biased fission contribution among a set of particle clusters and parameter changes"""
from single_component_sensitivity_analysis import single_component_sensitivity_analysis



if __name__ == "__main__":
    import sys
    import itertools
    from multiprocessing import Pool

    experiment_dir = [sys.argv[-1]]

    # Args for run for contrib of parameters =====================
    nSimulations = 100
    nClusters = [6]
    diffusion_trigger = [True]
    ffi_trigger = [True]
    targFis_probability_vals = [0.0,0.2,0.4,0.5,0.6,0.8,1.0]

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
