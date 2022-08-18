import argparse
from email.policy import default
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
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
from tensorflow.keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model_transformer import GraphConvolution, Time2Vec, AttentionBlock, positional_encoding
from utils import *

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--version', default="v1_1")
    parser.add_argument('--folder', default="V1")
    parser.add_argument('-f','--nfeat', type=int, default=90)
    parser.add_argument('-s','--seed', type=int, default=0)
    parser.add_argument('--support', default=1)
    parser.add_argument('--house_size', type = int, default=1600)
    parser.add_argument('--dropout', type = int, default=0.25)
    parser.add_argument('--head_size', type = int, default= 16)
    parser.add_argument('--trans_block',type=int, default=1)
    parser.add_argument('--ff_dim', type=int, default = 1080)
    parser.add_argument('--num_heads', type=int, default = 8)
    parser.add_argument('-p', '--patience', type=int, default=50)


    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    
    wait = 0
    preds = None
    best_val_mse = 9999999
    NB_EPOCH = 1000
    PATIENCE = args.patience

    X = np.load("/content/drive/Othercomputers/My_laptop/MasterArbeit/train_model/" + args.folder + "/feature_matrix_"+ args.version+".npy")
    X = MinMaxScaler().fit_transform(X)
    D = sp.load_npz("/content/drive/Othercomputers/My_laptop/MasterArbeit/train_model/" + args.folder + "/adjacency_matrix_" + args.version + ".npz")
    S = sp.load_npz("/content/drive/Othercomputers/My_laptop/MasterArbeit/train_model/" + args.folder + "/status_matrix_" + args.version + ".npz")
    y = np.load("/content/drive/Othercomputers/My_laptop/MasterArbeit/train_model/" + args.folder + "/label_nodes_"+ args.version + ".npy")/100000
    A = D.multiply(S)
    A_ = preprocess_adj(A, True)
    graph = [X, A_]
    G = [Input(shape=(None,None), sparse=True)]

    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

    whole_house = X.shape[0]
    house_size = args.house_size
    seq_len = int(whole_house/house_size)
    dropout = args.dropout
    head_size = args.head_size
    num_heads = args.num_heads
    ff_dim = args.ff_dim
    num_transformer_blocks = args.trans_block

    #Create Model
    X_in = Input(shape=(X.shape[1],))
    H = X_in
    H = GraphConvolution(args.nfeat, args.support, activation='relu')([H]+G)
    H = GraphConvolution(args.nfeat, args.support, activation='relu')([H]+G)
    H = Dropout(0.5)(H)
    H = Lambda(lambda H: tf.reshape(H, shape=(-1, H.shape[-1]), name="reshape_layer"))(H)
    seq_list = []
    for i in range(seq_len):
        seq_list.append(tf.gather(params=H, indices=range(i*house_size, (i+1)*house_size)))
    sequence = tf.stack(seq_list, 1)
    #positional encoding
    sequence *= tf.math.sqrt(tf.cast(args.nfeat, tf.float32))
    sequence += positional_encoding(sequence.shape[1], args.nfeat)[:, :tf.shape(sequence)[1],:]
    #transformer encoder
    for _ in range(num_transformer_blocks):
      C= AttentionBlock(head_size=head_size, num_heads = num_heads, ff_dim=ff_dim, dropout=dropout)(sequence)
    T = Dense(X.shape[1])(C)
    L = Lambda(lambda x : tf.keras.backend.reshape(x, (-1, args.nfeat)))(T)
    Y = Dense(1, activation = 'linear')(L)

    model = Model(inputs = [X_in]+G, outputs = Y)
    model.compile(optimizer=Adam(lr=0.001), loss = tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae'])
    
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

    with open('saved_model/gcn_transformer_ori_loss_{}'.format(args.version),'wb') as gcn_transformer_loss:
        pickle.dump(loss_history, gcn_transformer_loss, protocol=pickle.HIGHEST_PROTOCOL)

    model.save('saved_model/gcn_transformer_ori_model_{}'.format(args.version))

    plt.plot(range(0,len(all_val_mse)), all_val_mse, label = "Val Loss")
    plt.plot(range(0,len(all_train_mse)), all_train_mse, label = "Train Loss")
    plt.title('GCN Transformer Ori Model Loss {}'.format(args.version))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim([0,6])
    plt.legend()
    plt.savefig('saved_model/plot_gcn_transformer_ori_loss_{}'.format(args.version))