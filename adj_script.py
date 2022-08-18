from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from class_graph_dok import Graph
from math import radians, cos, sin, asin, sqrt
from scipy.spatial.distance import dice
from numpy import save
import scipy.sparse

def import_data(file_name):
    df = np.load(file_name)
    df = pd.DataFrame(df)
    return df

def sampling_data(dataframe, start, size):
    #len = dataframe.shape[0]
    end = int(start)+int(size)
    df_sample = dataframe[int(start):end]
    return df_sample

def take_status(dataframe):
    df = dataframe.copy()
    #drop non boolean features
    #90 : Price
    #91 : Transaction date
    #92 : Long
    #93 : Lat
    #94 : Group for time
    df = df.drop(df.iloc[:, 90:95], axis=1)
    #80 : Number of Residential Units
    #81 : Number of Commercial Units
    #82 : Number of Total Units
    #83 : Land Square Feet
    #84 : Gross Square Feet
    #85 : Year Built
    df = df.drop(df.iloc[:, 80:86], axis=1)
    #make sure it's boolean
    df[df<1] = 0
    df[df>1] = 1
    df = df.reset_index(drop=True)
    return df

def take_features(dataframe):
    dataframe = dataframe.reset_index(drop=True)
    all_feature = dataframe.iloc[:, :90]
    all_feature[all_feature<1]=0
    x_feature = dataframe.iloc[:,80:90]
    x_feature[x_feature<1]=0
    y_sample = dataframe.iloc[:, 90]
    coord = dataframe.loc[:,[93,92]]
    coord.columns=['lat','long']
    return all_feature, x_feature, y_sample, coord

def distance(lat1, lat2, lon1, lon2):
    #The math module contains a function named
    #radians which converts from degrees to radians
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    
    #Haversine formula
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    
    c = 2*asin(sqrt(a))
    
    #Radius of earth in kilometers. Use 3956 for miles
    r = 6371
    
    #Calculate the result
    return(c*r)

def adj_distance(df, threshold, threshold_dist):
    gr_update = Graph(df.shape[0])
    for idx, i in df.iterrows():
        range1_lat = i[0]-threshold
        range2_lat = i[0]+threshold
        range1_long = i[1]-threshold
        range2_long = i[1]+threshold
        val_lat = (df['lat'] > range1_lat) & (df['lat'] < range2_lat)
        val_long = (df['long'] > range1_long) & (df['long'] < range2_long)  
        val_true_index = np.logical_and(val_lat, val_long)
        val_index = val_true_index.index[val_true_index]
        for j in val_index:
            if j > idx:
                val_dist = distance(df.loc[idx][0], df.loc[j][0], df.loc[idx][1], df.loc[j][1])
                if (val_dist < threshold_dist) & (val_dist != 0):
                    gr_update.add_edge(idx, j , 1/val_dist)
    return gr_update

def dice_distance(df):
    gr = Graph(df.shape[0])
    for i, i_val in df.iterrows():
        i_val = np.array(i_val)
        for j , j_val in df.iterrows():
            j_val = np.array(j_val)
            dist = dice(i_val, j_val)
            gr.add_edge(i,j,1-dist)
    return gr

def save_file(df_input, output_file):
    scipy.sparse.save_npz(output_file, df_input)

def save_npy(df_input, output_file):
    save(output_file, df_input)

