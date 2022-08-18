# GCNTransformer
Code for paper Housing Price Prediction using Graph Convolutional Transformer Network

## Running

### Data Preprocessing
`python Sampling.py` to sample from the data, we need to specify :
- How many days is in one time window,
- How many nodes we would like to sample per time window.


`python main.py` to build adjacency matrix, similarity matrix, feature matrix, label nodes.

### Model Training
`python baseline.py`, to train using Regression.

`python gcn_only.py`, using only 2 GCN layers.

`python gcn_lstm.py`, 2 GCN layers + LSTM layer.\
We need to specify :
- Number of nodes per time window (`--house size`)

`python gcn_transformer_original.py`, 2 GCN layers with Transformer encoder + sinusoidal positional embedding[^1].\
We need to specify:
- Number of nodes per time window 
- Feed forward dimension

`python gcn_transformer.py`, 2 GCN layers with Transformer encoder + Time2Vec[^2] as time embedding.\
We need to specify:
- Number of nodes per time window 
- Feed forward dimension


[^1]: From the paper [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
[^2]: From the paper [Time2Vec: Learning a Vector Representation of Time](https://arxiv.org/abs/1907.05321)
