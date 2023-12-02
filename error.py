from normalize import Normalize
import matplotlib.pyplot as plt

class Error:
    """ Class finds error on intensity of normalized absorption spectrum through analyzing baseline noise
    """
    meting = Normalize(6, 1)
    x,y = meting.isolate(550, 575k)
    plt.figure()
    plt.plot(x,y)