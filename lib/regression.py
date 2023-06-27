import numpy as np
from scipy.signal import savgol_filter


class LinearFunction:
    def __init__(self, slope: float, intercept: float) -> None:
        """Generate a simple linear function given a slope and intercept

        Args:
            slope (float): slope of line
            intercept (float): intercept of line
        """
        self.slope = slope 
        self.intercept = intercept

    def __call__(self, input_x: float) -> float:
        """Returns the Y value of the linear function given an X

        Args:
            input_x (float): input x value

        Returns:
            float: output Y value
        """
        return self.slope * input_x + self.intercept


class ReLinearFunction(LinearFunction):
    """Difference between this class and its parent class is that
    output is forced to be non-negative.
    Given an X value, if Y computes to be negative, it is set to 0 instead.
    """
    def __init__(self, slope: float, intercept: float) -> None:
        """Call on super LinearFunction to populate slope and intercept values

        Args:
            slope (float): slope of line
            intercept (float): intercept of line
        """
        super().__init__(slope, intercept)

    def __call__(self, input_x: float) -> float:
        """Returns the Y value of the linear function given an X

        Args:
            input_x (float): input x value

        Returns:
            float: output Y value
        """
        out_val = super().__call__(input_x)
        if out_val < 0:
            return 0
        return out_val


class LinearPoints(LinearFunction):
    def __init__(self, pt1, pt2) -> None:
        """Generates a linear function given two points of a line

        Args:
            pt1 (np.ndarray): points in [x,y] format
            pt2 (np.ndarray): points in [x,y] format
        """
        self.pt1_x, self.pt1_y = pt1
        self.pt2_x, self.pt2_y = pt2

        self.slope = (self.pt2_y-self.pt1_y)/(self.pt2_x-self.pt1_x)
        self.intercept = self.pt2_y - (self.slope*self.pt2_x)

    def __call__(self, input_x: float) -> float:
        """calculate value of linear line at input_x

        Args:
            input_x (float): x_value

        Returns:
            float: y value at x_value
        """
        return super().__call__(input_x)


class ExponentialFunction:
    def __init__(self, m: float, t: float) -> None:
        """Generate an exponential function given a factor m and power t

        Args:
            m (float): coefficient constant
            t (float): exponent constant
        """
        self.m = m
        self.t = t

    def __call__(self, input_x: float) -> float:
        """Returns the Y value of the exponential function given an X

        Args:
            input_x (float): input x value

        Returns:
            float: output Y value
        """
        return self.m * np.exp(-self.t * (input_x))


class DataBasedFunction:
    def __init__(self, x: np.array, y: np.array, no_negatives: bool=False) -> None:
        """Generates a data based piecewise function made of multiple linear segments
        after data is passed through a linear savgol filter

        Args:
            x (np.array): input x vector based on data collected
            y (np.array): input y vector based on data collected
            no_negatives (bool): whether to bump negative values to zero
            
        """
        self.x = x
        self.y = y
        self.no_negatives = no_negatives

    def __call__(self, input_x: float) -> float:
        """Calculate the value of the function at x_value based on linear interpolation of 
        closest points in the dataset. If x_value is less than the min value in self.x, then 
        the closest two x values and their associated y values are used to calculate the interpolated 
        y value.  Also applies to max value.

        Args:
            input_x (float): x value 

        Returns:
            float: y value at input_x
        """
        differences = np.abs(self.x - input_x)
        idx = np.argpartition(differences, 2)
        closest, next_closest = idx[:2]

        pt0 = np.array([self.x[closest], self.y[closest]])
        pt1 = np.array([self.x[next_closest], self.y[next_closest]])
        func = LinearPoints(pt0, pt1)
        eval = func(input_x)

        # Added hard floor; since this function is being used to evaluate diffusion and probability values,
        # neither can be negative
        if self.no_negatives:
            if eval < 0:
                return 0
        return eval


def ExponentialFit(x, m, t):
    ''' Exponential function for fit'''
    return m * np.exp(-t * (x))


