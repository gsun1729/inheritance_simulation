import numpy as np
from typing import List
from copy import deepcopy


def time_alignment(reference_timestamps: np.ndarray,
                   recorded_timestamps: List[float],
                   precision: int = 2) -> List[float]:
    """Since timestamping in the simulation is prone to floating point errors where
    timestamps like 0.1 might be recorded as 0.10000000005, this function takes a list
    of desired reference timestamps and compares it against a list of recorded timestamps.
    The closest value in time between the two is then recorded as the correct timestamp 
    and a new list of reference timestamps is generated.

    Because the simulation only records network states when there is a physical change in the simulation, 
    some timestamps are not recorded, and the best reference case for simulation structural state is 
    the previous timestamp before any network change (remodeling) has been executed.

    For example, if the reference_timestamps are :
        [0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0]
    And the recorded timestamps are 
        [0.1,
        0.2000000005,
        0.4000000003,
        0.6000000006,
        0.7,
        0.90005,
        1.0]

    The function output will be 
        [0.1,
        0.1,
        0.2000000005,
        0.2000000005,
        0.4000000003,
        0.4000000003,
        0.6000000006,
        0.7,
        0.7
        0.90005]

    Function traverses every pair progressive of instances in recorded_timstamps 
    (referenced as lower, and upper), takes all values in reference_timstamps that are
    bounded by the lower (inclusive) and upper(exclusive), and sets them equal to the lower value.

    Args:
        reference_timestamps (np.ndarray): list of desired reference timestamps
        recorded_timestamps (List[float]): list of timestamps recorded by the simulation
        precision (int, optional): precision to round recorded timestamps to for comparison step. 
        Defaults to 2.

    Returns:
        List[float]: List of values sampled from recorded_timestamps.
    """
    new_reference_ts = deepcopy(reference_timestamps)
    recorded_timestamps.sort()
    for lower, upper in zip(recorded_timestamps[:-1], recorded_timestamps[1:]):
        # Mask reference time around lower(inclusive) and upper bound (exclusive)
        new_reference_ts[(new_reference_ts >= np.around(lower, decimals=precision)) & (
            new_reference_ts < np.around(upper, decimals=precision))] = lower
    new_reference_ts = list(new_reference_ts)
    return new_reference_ts
