"""This script is intended to run on simulation saved state files that have both 
PRE and POST states present, with IDs formatted as per README standard.
If the filenames are changed,id_lookup_len will have to be uopdated in the Experiment class """
import os
from lib.analytics import Experiment
from configs.inheritance_configs import CHAIN_LENGTH

if __name__ == "__main__":
    import sys

    PATH = sys.argv[-1]
    prefix = os.path.basename(PATH)
    prefix = prefix.replace(".", "_")
    
    a = Experiment(PATH, [".pickle"], id_lookup_len=7)
    f = a.batch_getInheritedEndAttr(limitLen=CHAIN_LENGTH)
    f.to_csv(f"{prefix}_batch_getInheritedEndAttr.csv")