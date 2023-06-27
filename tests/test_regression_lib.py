import matplotlib.pyplot as plt
import numpy as np
from lib.regression import (LinearPoints, DataBasedFunction)

if __name__ == "__main__":
    pt0 = np.array([0, 0])
    pt1 = np.array([1, 1])

    x = np.linspace(0.01, 2, 50)
    y = np.log(x)

    a = LinearPoints(pt0, pt1)
    b = DataBasedFunction(x, y)
    # Test data based function flooring for negative values
    c = DataBasedFunction(x, y, no_negatives=True)
    
    x = np.linspace(0.01, 5, 100)
    y = np.log(x)
    y2 = np.array([a(i) for i in x])
    y3 = np.array([b(i) for i in x])
    y4 = np.array([c(i) for i in x])
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(x, y4)
    plt.show()
