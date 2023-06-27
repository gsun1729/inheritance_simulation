from lib.read_utils import fromPickle2LinearFunction, fromPickle2RELinearFunction


DELTA_T = 0.1  # s

# These functions should take in a file with units of:
# for X values : volume (um3)
# for Y values : Diffusion constant (um2/s)

f_wt_diffConstVsVol = fromPickle2RELinearFunction(
    "config_files/y200_wt_D_vs_vol.pickle")

f_DUMPneg_diffConstVsVol = fromPickle2RELinearFunction(
    "config_files/y236_DUMPneg_D_vs_vol.pickle")

f_DUMPpos_diffConstVsVol = fromPickle2RELinearFunction(
    "config_files/y236_DUMPpos_D_vs_vol.pickle")

