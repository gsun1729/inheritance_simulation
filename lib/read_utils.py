import pickle
from lib.regression import LinearFunction, DataBasedFunction, ReLinearFunction


def readPickleFile(filepath: str):
    """Given a path to a pickle file containing a python data structure, 
    read in the pickled file.

    Args:
        filepath (str): path to the pickle file

    Returns:
        _type_: varies depending on contents of serialized data.
    """
    with open(filepath, "rb") as input_file:
        data = pickle.load(input_file)
    return data


def fromPickle2LinearFunction(pickle_path: str) -> LinearFunction:
    """Given a pickle path containing information on the slope and 
    intercept of a linear line, populate a LinearFunction with the
    values included in the picle file

    Args:
        pickle_path (str): path to the pickle file containing
            dict with keys "slope" and "intercept"

    Returns:
        LinearFunction: Linear function 
    """
    with open(pickle_path, "rb") as input_file:
        data = pickle.load(input_file)
    func = LinearFunction(slope=data.get('slope'),
                          intercept=data.get('intercept'))
    return func


def fromPickle2RELinearFunction(pickle_path: str) -> ReLinearFunction:
    """Given a pickle path containing information on the slope and 
    intercept of a linear line, populate a ReLinearFunction with the
    values included in the picle file

    Args:
        pickle_path (str): path to the pickle file containing
            dict with keys "slope" and "intercept"

    Returns:
        ReLinearFunction: Linear function 
    """
    with open(pickle_path, "rb") as input_file:
        data = pickle.load(input_file)
    func = ReLinearFunction(slope=data.get('slope'),
                            intercept=data.get('intercept'))
    return func


def fromPickle2DataBasedFunction(pickle_path: str) -> DataBasedFunction:
    """Given a pickle path containing 2D experimental data with 
    attributes x and y, plug the data into a DataBasedFunction class
    to use as a function

    Args:
        pickle_path (str): path to the pickle file containing
            dict with keys "x" and "y" for independent variable data
            on the x axis, dependent variable data on the y axis

    Returns:
        ReLinearFunction: Linear function 
    """
    with open(pickle_path, "rb") as input_file:
        data = pickle.load(input_file)
    func = DataBasedFunction(x=data.get('x'), 
                             y=data.get('y'))
    return func
