import argparse

from more_itertools import sample
from yaml import parse
from adj_script import dice_distance, import_data, sampling_data, save_file, adj_distance, take_features, take_status, save_npy

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default="housing_normal.npy")
    parser.add_argument('-oam','--output_adj_matrix', default="adjacency_matrix.npz")
    parser.add_argument('-osm', '--output_stat_matrix', default="status_matrix.npz")
    parser.add_argument('-fm', '--feature_matrix', default="feature_matrix.npy")
    #parser.add_argument('-xf', '--x_feature', default="x_feature.npy")
    parser.add_argument('-ln', '--label_nodes', default="label_nodes.npy")
    #parser.add_argument('-co', '--output_coords', default="coords.npy")
    #parser.add_argument('-st', '--stat_data', default="stat_data.npy")
    #parser.add_argument('-ol','--output_list', default="adjacency_list.npy")
    parser.add_argument('-t', '--threshold', default=0.05)
    parser.add_argument('-td', '--threshold_dist', default=5)
    parser.add_argument('--start', default=0)
    parser.add_argument('-s', '--size', type= int, default=9600)

    args = parser.parse_args()

    df_input = import_data(args.data)
    sample_data = sampling_data(df_input, args.start, args.size)
    status_data = take_status(sample_data)

    #calculate similarity
    status_mat = dice_distance(status_data)
    status_matrix = status_mat.to_csr()

    all_features, x_feature, label_nodes, coord = take_features(sample_data)
    
    #calculate distance
    adj_mat = adj_distance(coord, args.threshold, args.threshold_dist)
    adj_matrix = adj_mat.to_csr()
    # coord = coord.to_numpy()
    # stat = status_data.to_numpy()


    #save_npy(coord, args.output_coords)
    save_file(adj_matrix, args.output_adj_matrix)
    save_file(status_matrix, args.output_stat_matrix)
    save_npy(all_features, args.feature_matrix)
    #save_npy(x_feature, args.x_feature)
    save_npy(label_nodes, args.label_nodes)
    #save_npy(stat, args.stat_data)
