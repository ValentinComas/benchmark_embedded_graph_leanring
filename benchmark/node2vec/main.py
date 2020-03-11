from node2vec import Node2Vec
import numpy as np
import random
import networkx as nx
from IPython.display import Image
import matplotlib.pyplot as plt
import sys
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import node2vec
from utils import featToArray, save_order
import argparse
import random
import torch
import torch.nn as nn
import time
from ggnn import GatedGraphNN
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "datasets/fb_0/"
MODELS_DIR = "models/"
EMBEDDINGS_DIR = "embeddings/"
ORDERS_DIR = "orders/"
DATASET_NAME = "0"

def create_mapping_string_to_int(min_index="0", max_index="0"):
    mapping_dict = {str(x):x for x in range(min_index,max_index)}
    return mapping_dict

def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80, with_labels = False)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")

def mapper(vectors, position_map, pos):
    # print(torch.FloatTensor(vectors[int(position_map[int(pos)])]))
    return torch.FloatTensor(vectors[int(position_map[int(pos)])])

def process_data(data, vectors, position_map,device):
    train_data, train_labels = data[:,:-1], data[:,-1]
    traited_train_data = torch.empty(len(train_data),2,vectors.shape[1])
    for i in range(len(train_data)):
        traited_train_data[i][0] = mapper(vectors, position_map, train_data[i][0])
        traited_train_data[i][1] = mapper(vectors, position_map, train_data[i][1])
    traited_train_labels = torch.empty(len(train_labels),2)
    # traited_train_labels = torch.FloatTensor([1. if i == '1' else 0. for i in train_labels])
    for i in range(len(train_labels)):
        if train_labels[i] == '1':
            traited_train_labels[i] = torch.FloatTensor([0.,1.])
        elif train_labels[i] == '0':
            traited_train_labels[i] = torch.FloatTensor([1.,0.])
    return traited_train_data, traited_train_labels

def generate_node2vec():
    n = 34
    m = 78
    G = nx.read_edgelist(DATA_DIR + DATASET_NAME + ".edges")
    feat_list = featToArray(DATA_DIR + DATASET_NAME)
    for line in feat_list:
        if not(G.has_node(line[0])):
            G.add_node(line[0])
        for i in range(1, len(line)):
            G.nodes[line[0]][i-1] = line[i]
    save_order(G.nodes, ORDERS_DIR + DATASET_NAME + ".txt")
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=500, workers=8)

    # Embed
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    model.wv.save_word2vec_format(EMBEDDINGS_DIR + DATASET_NAME + ".emb")
    model.save(MODELS_DIR + DATASET_NAME + ".model")

def generate_links_data(fake_part, training_part):
    f = open(DATA_DIR + DATASET_NAME + ".edges", "r")
    G = nx.read_edgelist(DATA_DIR + DATASET_NAME + ".edges")
    links = {(i.split(' ')[0], i.split(' ')[1].replace('\n', '')):1 for i in f.readlines()}
    fake_num = (int)(len(links) * fake_part / (1-fake_part))
    for i in range(fake_num):
        node_1 = random.randint(1, len(G.nodes))
        node_2 = random.randint(1, len(G.nodes))
        while (node_1, node_2) in links or (node_2, node_1) in links:
            node_1 = random.randint(1, len(G.nodes))
            node_2 = random.randint(1, len(G.nodes))
        links[(node_1, node_2)] = 0
    training_data = {}
    for i in range((int)(training_part * len(links))):
        c_key = random.choice(list(links.keys()))
        training_data[c_key] = links[c_key]
        del links[c_key]
    f = open('prepared_data/training_fb_0.txt', 'w')
    for n1, n2 in list(training_data.keys()):
        f.write(str(n1) + ' ' + str(n2) + ' ' + str(training_data[(n1,n2)]) + '\n')
    f.close()
    f = open('prepared_data/test_fb_0.txt', 'w')
    for n1, n2 in list(links.keys()):
        f.write(str(n1) + ' ' + str(n2) + ' ' + str(links[(n1,n2)]) + '\n')
    f.close()

def train(rnn_type): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = Word2Vec.load(MODELS_DIR + DATASET_NAME + ".model")
    f = open(ORDERS_DIR + DATASET_NAME + ".txt", "r")
    j=1
    position_map = {}
    for i in f.readlines():
        position_map[j] = i.replace('\n', '')
        j+=1
    f.close()
    data = []
    f = open('prepared_data/training_fb_0.txt', 'r')
    data = np.array([[n for n in line.replace('\n', '') .split(' ')] for line in f.readlines()])
    f.close()
    train_data, train_labels = process_data(data, model.wv.vectors, position_map, device)
    # print(train_labels)
    batch_size = 256
    train_d = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_d, shuffle=True, batch_size=batch_size, drop_last=True)
    input_size = next(iter(train_loader))[0].shape[2]
    output_size = 2
    model_nn = GatedGraphNN(rnn_type, input_size, 64,output_size, 2, False)
    model_nn.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), parser.lr)
    epochs = parser.epochs
    epoch_times = []
    # print(train_loader)

    model_nn.train()
    for epoch in range(1,epochs+1):
        start_time = time.clock()
        h = model_nn.init_hidden(batch_size, device)
        avg_loss = 0.
        counter = 0
        for data, label in train_loader:
            # if epoch == 1 and counter == 0:
            #     print("hello")
            #     print(label)
            counter += 1
            if rnn_type == 'gru':
                h = h.data
            elif rnn_type == 'lstm':
                h = tuple([i.data for i in h])
            model_nn.zero_grad()
            out, h = model_nn(data.to(device), h)
            # print(out)
            # if epoch == epochs:
            #     print(out)
            # print(out)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%100 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, parser.epochs, avg_loss/len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    torch.save(model_nn, rnn_type + '.model')

        # X_embedded = TSNE(n_components=2).fit_transform(model.wv.vectors)
        
        # alpha=0.9
        # label_map = { l: i for i, l in enumerate(np.unique(X_embedded))}
        

        # rng = np.random.RandomState(0)
        # colors = rng.rand(333)
        # plt.figure(figsize=(10,8))
        # plt.scatter(X_embedded[:,0], 
        #             X_embedded[:,1], 
        #             cmap="jet", alpha=alpha)

        # plt.show()

def test(rnn_type):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = Word2Vec.load(MODELS_DIR + DATASET_NAME + ".model")
    f = open(ORDERS_DIR + DATASET_NAME + ".txt", "r")
    j=1
    position_map = {}
    for i in f.readlines():
        position_map[j] = i.replace('\n', '')
        j+=1
    f.close()
    data = []
    f = open('prepared_data/test_fb_0.txt', 'r')
    data = np.array([[n for n in line.replace('\n', '') .split(' ')] for line in f.readlines()])
    f.close()
    test_data, test_labels = process_data(data, model.wv.vectors, position_map, device)
    test_d = TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_d, batch_size=1, shuffle=False)


    model_nn = torch.load(rnn_type + '.model')
    model_nn.to(device)
    corrects = 0
    model_nn.eval()
    
    for data, label in test_loader: 
        inp = torch.from_numpy(np.array(data))
        h = model_nn.init_hidden(inp.shape[0], device)
        if rnn_type == 'gru':
            h = h.data
        out, h = model_nn(data.to(device).float(), h)
        if torch.argmax(out) == torch.argmax(label):
            corrects += 1
        # if out > 0.5 and label == 1. or out < 0.5 and label == 0.:
        #     corrects +=1
    print(corrects/len(test_d))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mode', default=0.001)                                 # Choose train or model
    # parser.add_argument('-dataset_num', default='')                           # Choose train or model
    # parser.add_argument('-dataset_name', default='')                          # Choose train or model
    parser.add_argument('-lr', default=0.001)                                   # Learning rate
    parser.add_argument('-hidden_size', type=int, default=256)                  # Size of the GRU / Memory cell
    parser.add_argument('-num_layers',type=int, default=2)                      # Number of layers of the GRU / Memory cell
    parser.add_argument('-epochs', type=int, default=8)                         # Epochs
    parser.add_argument('-b', '--batch_size', type=int, default=1024)           # Batch_size
    parser.add_argument('-bi', '--bidirectionnal', type=int, default=False)     # Wether the GRU / Memory Cell is bidirectionnal or not
    parser.add_argument('-rnn', default='gru')                        # Wether it is GRU or LSTM is bidirectionnal or not
    
    parser = parser.parse_args()
    
    if parser.mode == "generate_node2vec":
        generate_node2vec()
    elif parser.mode == "generate_links_data":
        generate_links_data(0.5,0.5)
    elif parser.mode == "train":
        train(parser.rnn)
    elif parser.mode == "test":
        test(parser.rnn)