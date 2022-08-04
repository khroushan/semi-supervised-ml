import numpy as np
import pandas as pd
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
    def shuffle(xy: list):
        """ To shuffle data and keep the x and y in the same order """
        # shuffle both x and y
        p = np.random.permutation(xy[0].shape[0])
        return (s[p] for s in xy)

    def mask(self, u_frac):
        """ Mask part of the data as unlabeled, update data """
        if (u_frac < 0.) | (u_frac > 1.0):
            raise Exception("Error: u_frac must be between 0 and 1")
        x, y = self.shuffle([self.x, self.y])
        # determine number of unlabeled datapoints.
        u_num = int(y.shape[0]*u_frac)
        x_u, y_u = x[:u_num], y[:u_num]
        x_l, y_l = x[u_num:], y[u_num:]
        # mask labeled to -1
        y_u_mask = -1*np.ones(y_u.shape)
        # convert to proper sparse matrix and reshuffle
        x, y, y_mask = sparse.vstack([x_u, x_l]).tocsr(), \
            np.r_[y_u, y_l], \
            np.r_[y_u_mask, y_l]
        self.x = x
        self.y = y
        self.y_mask = y_mask

    def sample(self, l_frac, u_frac):
        """ Return a fraction of the data for labeled and unlabeled data
        points """
        x, y, y_mask = self.shuffle([self.x, self.y, self.y_mask])
        mask_indx = (y_mask == -1)
        x_u, y_u, y_u_mask = x[mask_indx], y[mask_indx], y_mask[mask_indx]
        x_l, y_l, y_l_mask = x[~mask_indx], y[~mask_indx], y_mask[~mask_indx]

        u_num = int(y_u.shape[0]*u_frac)
        l_num = int(y_l.shape[0]*l_frac)

        # sample of each array accordingly
        x_u_s, y_u_s, y_u_mask_s = x_u[:u_num], y_u[:u_num], y_u_mask[:u_num]
        x_l_s, y_l_s, y_l_mask_s = x_l[:l_num], y_l[:l_num], y_l_mask[:l_num]

        return sparse.vstack([x_u_s, x_l_s]),\
            np.r_[y_u_s, y_l_s],\
            np.r_[y_u_mask_s, y_l_mask_s]
