from lib.read_utils import fromPickle2RELinearFunction

# Functions take in Volume and output probabilities
f_P_wt_fission = fromPickle2RELinearFunction(
    "config_files/y200_wt_pPostOpFission_vs_vol.pickle")
f_P_wt_fusion = fromPickle2RELinearFunction(
    "config_files/y200_wt_pPostOpFusion_vs_vol.pickle")


f_P_dumpNeg_fission = fromPickle2RELinearFunction(
    "config_files/y236_DUMPneg_pPostOpFission_vs_vol.pickle")
f_P_dumpNeg_fusion = fromPickle2RELinearFunction(
    "config_files/y236_DUMPneg_pPostOpFusion_vs_vol.pickle")


f_P_dumpPos_fission = fromPickle2RELinearFunction(
    "config_files/y236_DUMPpos_pPostOpFission_vs_vol.pickle")
f_P_dumpPos_fusion = fromPickle2RELinearFunction(
    "config_files/y236_DUMPpos_pPostOpFusion_vs_vol.pickle")
