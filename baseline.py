import argparse
from cgi import test
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from more_itertools import sample
from yaml import parse
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

from model import Baseline

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', default="300_samples")
    parser.add_argument('-v', '--version', default="v1_1")
    parser.add_argument('-f','--nfeat', type=int, default=90)
    parser.add_argument('-s','--seed', type=int, default=0)

    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    X = np.load(args.folder + "/feature_matrix_"+ args.version +".npy")
    X = MinMaxScaler().fit_transform(X)
    y = np.load(args.folder + "/label_nodes_"+ args.version + ".npy")/100000

    data_df = pd.DataFrame(X)
    data_df['label'] = y

    train_dataset = data_df[:8000]
    train_dataset = train_dataset.drop('label', axis=1)
    test_dataset = data_df.drop(train_dataset.index)
    test_dataset = test_dataset.drop('label', axis=1)

    #Take index to list
    train_index = train_dataset.index.tolist()
    test_index = test_dataset.index.tolist()

    #Take train test dataset as dataframe
    train_label = pd.DataFrame(y[train_dataset.index], index=train_dataset.index)
    test_label = pd.DataFrame(y[test_dataset.index], index=test_dataset.index)

    model = Baseline(nfeat=args.nfeat)
    model.compile(optimizer='Adam', loss = 'mse', metrics=['mse','mae'])

    #Train
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=20)

    history = model.fit(train_dataset, train_label, epochs=1000, validation_split=0.17, callbacks=es)

    loss, mse, mae  = model.evaluate(test_dataset, test_label)

    pred = model.predict(test_dataset)
    pred = np.reshape(pred, len(pred))
    test_label = test_label.to_numpy()
    test_label = np.reshape(test_label, len(test_label))
    print(pred)
    print(test_label)
    pearson_corr = stats.pearsonr(pred, test_label)

    with open(args.folder + '/saved_model/baseline_loss_{}'.format(args.version),'wb') as baseline_history:
        pickle.dump(history.history, baseline_history)

    model.save(args.folder + '/saved_model/baseline_model_{}'.format(args.version))

    #print(X.shape)
    print('Test Data MSE:', mse)
    print('Test Data MAE:', mae)
    print('Pearson Correlation: ', pearson_corr)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim([0,10])
    plt.legend(['Train', 'Test'], loc = 'upper right')
    plt.savefig(args.folder + '/saved_model/plot_baseline_loss_{}'.format(args.version))