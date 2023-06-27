"""
Run this script to run most analyses in lib.analytics on a defined directory of pickle files.
NOTE: pickle files must be in the correct format, with id_lookup_len set to 7 elements.
WARNING: Running this script with a lot of experiments in your experiments directory will take a long time, 
    it is better to pick which analyses to run and run them individually (comment out un-run analyses)

"""
import os
import sys
from lib.analytics import Experiment
from configs.inheritance_configs import CHAIN_LENGTH


if __name__ == "__main__":
    

    PATH = sys.argv[-1]
    prefix = os.path.basename(PATH)
    prefix = prefix.replace(".", "_")

    a = Experiment(PATH, [".pickle"], id_lookup_len=7)
    
    print("Running batch_getAggReductionMitoEvents")
    f = a.batch_getAggReductionMitoEvents()
    f.to_csv(f"{prefix}_batch_getAggReductionMitoEvents.csv")
    
    
    start_time = 600
    stop_time = 0
    interval = 10
    duration = 300
    while stop_time <=1000:
        stop_time = start_time + duration
        print(start_time, stop_time)

        f = a.batch_getEventFrequency(start_timestamp=start_time,
                                      end_timestamp=stop_time,
                                      time_resolution=interval)
        f.to_csv(f"{prefix}_batch_getEventFrequency_{start_time}to{stop_time}.csv")

        f = a.batch_getVolumesAtTime(start_time)
        f.to_csv(f"{prefix}_batch_getVolumesAtTime{start_time}.csv")
        start_time += 100

    print("Running batch_getDeltaAggInterSep")
    f = a.batch_getDeltaAggInterSep()
    f.to_csv(f"{prefix}_batch_getDeltaAggInterSep.csv")

    print("Running batch_getEndNAggClusters")
    f = a.batch_getEndNAggClusters()
    f.to_csv(f"{prefix}_batch_getEndNAggClusters.csv")

    print("Running batch_getFirstAggFusionEventTime")
    f = a.batch_getFirstAggFusionEventTime()
    f.to_csv(f"{prefix}_batch_getFirstAggFusionEventTime.csv")

    print("Running batch_getInheritanceOverTime")
    f = a.batch_getInheritanceOverTime()
    f.to_csv(f"{prefix}_batch_getInheritanceOverTime_L{CHAIN_LENGTH}.csv")

    print("Running batch_getInheritedEndAttr")
    f = a.batch_getInheritedEndAttr()
    f.to_csv(f"{prefix}_batch_getInheritedEndAttr_L{CHAIN_LENGTH}.csv")

    print("Running batch_getNAggsOverTime")
    f = a.batch_getNAggsOverTime()
    f.to_csv(f"{prefix}_batch_getNAggsOverTime.csv")

    print("Running batch_getNSubnetsOverTime")
    f = a.batch_getNSubnetsOverTime()
    f.to_csv(f"{prefix}_batch_getNSubnetsOverTime.csv")

    print("Running batch_getVolumesOverTime")
    f = a.batch_getVolumesOverTime()
    f.to_csv(f"{prefix}_batch_getVolumesOverTime.csv")
