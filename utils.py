import numpy as np
import scipy.sparse as sp
import math
import tensorflow as tf


from sklearn.metrics import mean_squared_error, mean_absolute_error

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_splits(y):
    idx_train = range(6400)
    idx_val = range(6400,7999)
    idx_test = range(8000, 9599)
    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

def root_mse(preds, labels):
    return math.sqrt(mean_squared_error(labels, preds))

def pred_error_baseline(preds, labels):
    preds = preds.reshape(1,-1)
    labels = labels.reshape(1,-1)
    diff = np.fabs(preds - labels)
    pred_error = np.true_divide(diff, labels)
    pred_error = tf.convert_to_tensor(pred_error)
    #avg_pred_error = np.mean(pred_error)
    return tf.keras.backend.mean(pred_error)

def evaluate_preds_baseline(preds, labels):
    mse_loss = list()
    predict_error = list()
    mse_loss.append(root_mse(preds, labels))
    predict_error.append(pred_error(preds, labels))
    return mse_loss, predict_error

def pred_error(preds, labels):
    preds = preds.reshape(1,-1)
    labels = labels.reshape(1,-1)
    diff = np.fabs(preds - labels)
    pred_error = np.true_divide(diff, labels)
    avg_pred_error = np.mean(pred_error)
    return avg_pred_error

def evaluate_preds(preds, labels, indices):
    rmse_loss = list()
    predict_error = list()
    MAE = list()
    for y_split, idx_split in zip(labels, indices):
        rmse_loss.append(root_mse(preds[idx_split], y_split[idx_split]))
        predict_error.append(pred_error(preds[idx_split], y_split[idx_split]))
        MAE.append(mean_absolute_error(y_split[idx_split], preds[idx_split]))
    return rmse_loss, predict_error, MAE


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def backend_reshape(x, out):
    return tf.keras.backend.reshape(x,(-1,out))

def year_built(x):
    if x <1850 :
        return "<1850"
    elif x >= 1850 and x < 1900:
        return "1850-1900"
    elif x >= 1900 and x < 1910:
        return "1900-1910"
    elif x >= 1910 and x < 1920:
        return "1910-1920"
    elif x >= 1920 and x < 1930:
        return "1920-1930"
    elif x >= 1930 and x < 1940:
        return "1930-1940"
    elif x >= 1940 and x < 1950:
        return "1940-1950"
    elif x >= 1950 and x < 1960:
        return "1950-1960"
    elif x >= 1960 and x < 1970:
        return "1960-1970"
    elif x >= 1970 and x < 1980:
        return "1970-1980"
    elif x >= 1980 and x < 1990:
        return "1980-1990"
    elif x >= 1990 and x < 2000:
        return "1990-2000"
    else :
        return ">2000"

def condition_residential(x):
    if x == 0 :
        return "nonresidential"
    elif x > 0 and x < 5:
        return "small_residential"
    elif x>= 5 and x <=10:
        return "smmed_residential"
    elif x>10 and x <=25 :
        return "medium_residential"
    else :
        return "large_residential"

def condition_commercial(x):
    if x == 0 :
        return "noncommercial"
    elif x > 0 and x < 5:
        return "small_commerce"
    elif x>= 5 and x <=10:
        return "smmed_commerce"
    elif x>10 and x <=25 :
        return "medium_commerce"
    else :
        return "large_commerce"
