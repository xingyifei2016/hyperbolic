import torch.nn as nn
import torch

class logInverse(nn.Module):
    """ 
    Performs magnitude max pooling
    Pools the input with largest magnitude over neighbors
    """

    def __init__(self, k):
        super(logInverse, self).__init__()
        self.k = k
        

    def __repr__(self):
        return 'LogInverse'

    def forward(self, x):
        """
        x: Input tensor
        Compute ln^k(1/(x+1e-7) + 1)
        """
        
        return torch.log( 1/x + 1) ** (-1 * self.k)
