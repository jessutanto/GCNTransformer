import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import pickle
import os
import math

from more_itertools import sample
from yaml import parse
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from scipy import stats
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model import GraphConvolution
from utils import *

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--version', default="v1_1")
    parser.add_argument('--folder', default="V1")
    parser.add_argument('-f','--nfeat', type=int, default=90)
    parser.add_argument('-s','--seed', type=int, default=0)
    parser.add_argument('--support', default=1)

    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    
    wait = 0
    preds = None
    best_val_mse = 9999999
    NB_EPOCH = 1000
    PATIENCE = 50

    X = np.load(args.folder + "/feature_matrix_"+ args.version+".npy")
    X = MinMaxScaler().fit_transform(X)
    D = sp.load_npz(args.folder + "/adjacency_matrix_" + args.version + ".npz")
    S = sp.load_npz(args.folder + "/status_matrix_" + args.version + ".npz")
    y = np.load(args.folder + "/label_nodes_"+ args.version + ".npy")/100000
    A = D.multiply(S)
    A_ = preprocess_adj(A, True)
    graph = [X, A_]
    G = [Input(shape=(None,None), sparse=True)]

    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

    #Create Model
    X_in = Input(shape=(X.shape[1],))
    H = X_in
    H = GraphConvolution(args.nfeat, args.support, activation='relu')([H]+G)
    H = GraphConvolution(args.nfeat, args.support, activation='relu')([H]+G)
    H = Dropout(0.5)(H)
    Y = Dense(1, activation = 'linear')(H)

    model = Model(inputs = [X_in]+G, outputs = Y)
    model.compile(optimizer=Adam(lr=0.001), loss = tf.keras.losses.MeanSquaredError(), metrics=['mse','mae'])
    
    #Loss list
    all_val_mse = list()
    all_train_mse = list()
    all_val_mae = list()
    all_train_mae = list()
    all_val_error = list()
    all_train_error = list()

    for epoch in range(1, NB_EPOCH+1):
        t = time.time()

        #Single training iteration (we mask nodes without labels for loss calculation)
        history = model.fit(graph, y_train, sample_weight=train_mask,
                    batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        #Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        #Train/Test Scores
        (Train_mse, Val_mse),(Train_error, Val_error), (Train_mae, Val_mae) = evaluate_preds(
            preds, [y_train, y_val], [idx_train, idx_val])
        all_train_mse.append(Train_mse)
        all_val_mse.append(Val_mse)
        all_train_mae.append(Train_mae)
        all_val_mae.append(Val_mae)
        all_train_error.append(Train_error)
        all_val_error.append(Val_error)
        print("Epoch: {:04d}".format(epoch),
            "train_RMSE= {:.4f}".format(Train_mse),
            "val_RMSE= {:.4f}".format(Val_mse),
            "train_MAE= {:.4f}".format(Train_mae),
            "val_MAE= {:.4f}".format(Val_mae),
            "train_GER= {:.4f}".format(Train_error),
            "val_GER= {:.4f}".format(Val_error),
            "time= {:.4f}".format(time.time() - t))

        #Early Stopping
        if Val_mse < best_val_mse:
            best_val_mse = Val_mse
            wait = 0
        else:
            print("Current Best Test MSE:", best_val_mse)
            if wait >= PATIENCE:
                print("Epoch {}: Early stopping".format(epoch))
                break
            wait += 1

    (Test_mse), (Test_error), (Test_mae) = evaluate_preds(preds, [y_test], [idx_test])

    #pearson correlation between predicted value vs ground truth
    pred = preds[idx_test]
    pred = np.reshape(pred, len(pred))
    test_label = y_test[idx_test]
    test_label = np.reshape(test_label, len(test_label))
    pearson_correlation = stats.pearsonr(pred, test_label)

    print('Test Data MSE:', Test_mse)
    print('Test Data MAE:', Test_mae)
    print('Pearson Correlation: ', pearson_correlation)

    loss_data = [all_train_mse, all_val_mse, all_train_mae, all_val_mae, all_train_error, all_val_error]
    loss_dict = ['Train_Loss(RMSE)', 'Val_Loss(RMSE)', 'Train_MAE', 'Val_MAE', 'Train_GER', 'Val_GER']
    zipObj = zip(loss_dict, loss_data)
    loss_history = dict(zipObj)


    with open(args.folder+'/saved_model/gcn_loss_{}'.format(args.version),'wb') as gcn_loss:
        pickle.dump(loss_history, gcn_loss, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(args.folder+'/saved_model/gcn_model_{}'.format(args.version))

    plt.plot(range(0,len(all_val_mse)), all_val_mse, label = "Val Loss")
    plt.plot(range(0,len(all_train_mse)), all_train_mse, label = "Train Loss")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim([0,6])
    plt.legend()
    plt.savefig(args.folder+'/saved_model/plot_gcn_loss_{}'.format(args.version))