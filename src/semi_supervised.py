import numpy as np
import pandas as pd
import scipy as scp
from scipy import sparse

class SemiSupervisedData():
    """ Prepare classification dataset for semi-supervised training. """
    def __init__(self, 
                 x, 
                 y: np.ndarray):
        self.x = x
        self.y = y
        self.num_features = x.shape[0]
        
    @classmethod
    def from_dataframe(cls, 
                       df: pd.DataFrame,
                       x_cols: list,
                       y_col: str):
        """ If the data given in pd.DataFrame form """
        return cls(df[x_cols].values, df[y_col].values)
    
    @staticmethod
    def shuffle_data(x, y):
        """ To shuffle data and keep the x and y in the same order """
        # shuffle both x and y
        p = np.random.permutation(y.shape[0])
        return x[p], y[p]

        
    
    def mask_data(self, u_frac):
        """ Mask part of the data as unlabeled """
        if (u_frac < 0.) | (u_frac > 1.0) :
            raise Exception("Error: u_frac must be between 0 and 1")
            
        x, y = self.shuffle_data(self.x, self.y)
        
        u_num = int(y.shape[0]*u_frac)
        
        x_u, y_u = x[:u_num], y[:u_num]
        x_l, y_l = x[u_num:], y[u_num:]
        
        return sparse.vstack([x_u, x_l]), np.r_[y_u, y_l]

    def sample(self, l_frac, u_frac):
        """ Return a fraction of the data for labeled and unlabeled data points """
        
        x ,y =  self.shuffle_data(self.x, self.y)

        x_u, y_u = x[y == -1 ], y[y == -1 ]
        x_l, y_l = x[y != -1 ], y[y != -1 ]

        u_num = int(y_u.shape[0]*u_frac)
        l_num = int(y_l.shape[0]*l_frac)

        x_u_sample, y_u_sample = x_u[:u_num], y_u[:u_num]
        x_l_sample, y_l_sample = x_l[:l_num], y_l[:l_num]

        return sparse.vstack([x_u_sample, x_l_sample]), np.r_[y_u_sample, y_l_sample]
