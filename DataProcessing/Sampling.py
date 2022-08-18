import argparse
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
from scipy.io import loadmat
from numpy import save

from utils import year_built, condition_residential, condition_commercial

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_sample', type=int, default=500)
    parser.add_argument('-c', '--cycle', type=int, default = 28)
    parser.add_argument('-v', '--version', type=int, default=1)

    args = parser.parse_args()

    normal_data = loadmat(r"housing_normal.mat")

    #Reverse to the original data
    X_ori = (normal_data['X']*normal_data['sd_data'])+normal_data['me_data']
    df_X = pd.DataFrame(X_ori)
    df_X = df_X.drop_duplicates()

    #Sample
    df_ori = df_X.copy()
    df_ori = df_ori.sort_values(by=[91], ascending=True)
    minim = df_ori[91].min()
    df_ori[91] = df_ori[91] - minim + 1
    df_ori['Group'] = df_ori[91]//args.cycle
    #df_ori[91].max()
    df_sorted = df_ori.groupby('Group').sample(n=args.n_sample, random_state=args.version)
    df_sorted = df_sorted.reset_index(drop=True)

    #One Hot Encode
    df_sorted['residential_status'] = df_sorted[80].apply(condition_residential)
    df_sorted['commercial_status'] = df_sorted[81].apply(condition_commercial)
    df_sorted['year_group'] = df_sorted[85].apply(year_built)

    df_sorted = pd.get_dummies(df_sorted, columns=['residential_status', 'commercial_status', 'year_group'])

    df_save = df_sorted.to_numpy()

    save('housing_normal_{}_v{}.npy'.format(args.n_sample, args.version), df_save)