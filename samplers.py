import numpy as np
from scipy.stats import norm


class BasicSampler():

    """
    Simple sampler that relies on the ask method
    of the optimizers
    """

    def __init__(self, sample_archive, thetas_archive, **kwargs):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        return

    def ask(self, pop_size, optimizer):
        return optimizer.ask(pop_size), 0, 0, 0, 0, 0
