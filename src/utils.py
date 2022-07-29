import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score, \
                            auc,\
                            f1_score, \
                            recall_score, \
                            precision_score, \
                            classification_report

def prepare_U_L_data(df, in_label_col, out_label_col, u_frac):
    """ Mask a fraction of labels to make unlabeled dataset 
        input:
        df: dataframe 
        l_frac: fraction of labeled dataset
        u_frac: fraction of unlabeled dataset
        in_label_col: str, name of input target column
        out_label_col: str, name of output target column
    """
    if (u_frac < 0.) | (u_frac > 1.0) :
        print("Error: u_frac must be between 0 and 1")
    df_u = df.sample(frac=u_frac).copy()
    df_l = df[~df.index.isin(df_u.index)].copy()
    print("Unlabeled data shape: ", df_u.shape)
    print("Labeled data shape: ", df_l.shape)
    df_u.loc[:, out_label_col] = -1
    df_l.loc[:, out_label_col] = df_l.loc[:, in_label_col]
    return pd.concat([df_u, df_l])

def _data_sampler_df(df: pd.DataFrame,
                     feature_cols,
                     target_col, 
                     l_frac=1.,
                     u_frac=0.1):
    """ Return fraction ofd data for label and unlabel dataset to study effect of 
        portion of labeled and unlabeled dataset on performance of a trained model. 
        
        input:
        df: dataframe with collection of both labeled and unlabeled data. The
        unlabeled
        data are specified with label -1
        l_frac: desired fraction of labeled dataset
        u_frac: desired fraction of unlabeled dataset
    """
    df_u = df[df[target_col] == -1 ]
    df_l = df[df[target_col] != -1 ]
    
    df_u_sample = df_u.sample(frac=u_frac)
    df_l_sample = df_l.sample(frac=l_frac)
    
    return pd.concat([df_u_sample, df_l_sample])

def evaluate_model(true_label, predicted_label):
    """ Classification reports """
    
    fs = f1_score(true_label, predicted_label)
    pc = precision_score(true_label, predicted_label)
    rc = recall_score(true_label, predicted_label)
    roc = roc_auc_score(true_label, predicted_label)

    return {'f1': fs, "precision":pc, "recall":rc, "roc":roc}

def self_trainer(ml_trainer, 
                 feature_cols, 
                 target_col, 
                 train,
                 test,
                 l_frac, 
                 u_frac, 
                 n_realization):
    fs = []; pc = []; rc = []; roc = []
    
    x_test, y_test = _test_handle(test, feature_cols, target_col)
        
    for i in range(n_realization): # loop on realizations
        x_train, y_train = data_sampler(train, 
                                        feature_cols, 
                                        target_col, 
                                        l_frac=l_frac, 
                                        u_frac=u_frac)
        ml_trainer.fit(x_train, y_train)
        test_prediction = ml_trainer.predict(x_test)
        
        metrics = evaluate_model(y_test, test_prediction)
        fs.append(metrics['f1'])
        pc.append(metrics['precision'])
        rc.append(metrics['recall'])
        roc.append(metrics['roc'])

    return np.mean(roc), np.std(roc)

def _test_handle(test, feature_cols, target_col):
    """ Return test features and target """
    if isinstance(test, pd.DataFrame):
        x, y = test[feature_cols], test[target_col]
    if isinstance(test, list):
        x, y = test[0], test[1]
    return x, y
                                                  
def _data_sampler_list(data: list,
                       l_frac=1.,
                       u_frac=0.1):
    """ Return fraction ofd data for label and unlabel dataset to study effect of 
        portion of labeled and unlabeled dataset on performance of a trained model. 
        
        input:
        df: dataframe with collection of both labeled and unlabeled data. The
        unlabeled
        data are specified with label -1
        l_frac: desired fraction of labeled dataset
        u_frac: desired fraction of unlabeled dataset
    """
    if not (sparse.issparse(data[0]) & isinstance(data[1], np.ndarray)):
        raise Exception("Two numpy arrays must be paassed!")
    
    x = data[0]
    y = data[1]
    # shuffle both x and y
    p = np.random.permutation(y.shape[0])
    x ,y =  x[p], y[p]
    
    x_u, y_u = x[y == -1 ], y[y == -1 ]
    x_l, y_l = x[y != -1 ], y[y != -1 ]
    
    u_num = int(y_u.shape[0]*u_frac)
    l_num = int(y_l.shape[0]*l_frac)
    
    x_u_sample, y_u_sample = x_u[:u_num], y_u[:u_num]
    x_l_sample, y_l_sample = x_l[:l_num], y_l[:l_num]
        
    return sparse.vstack([x_u_sample, x_l_sample]), np.r_[y_u_sample, y_l_sample]

def data_sampler(data,
                 feature_cols,
                 target_col,
                 l_frac=1.,
                 u_frac=0.1):
    if isinstance(data, pd.DataFrame):
        df = _data_sampler_df(df,
                              feature_cols,
                              target_col,
                              l_frac=l_frac,
                              u_frac=u_frac)
        x, y = df[feature_cols], df[target_col]
    if isinstance(data, list):
        x, y = _data_sampler_list(data,
                                  l_frac=l_frac,
                                  u_frac=u_frac)
        
    if not isinstance(data, (pd.DataFrame, list)):
        raise Exception("Passed data must be either pd.DataFrame or list of np.array")
        
    return x, y