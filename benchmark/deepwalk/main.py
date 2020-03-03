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

DATA_DIR = "datasets/fb_0/"
MODELS_DIR = "models/"
EMBEDDINGS_DIR = "embeddings/"
DATASET_NAME = "karate"

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


#it can also be saved in .svg, .png. or .ps formats

if __name__ == '__main__':
    if sys.argv[1] == "train":
        n=34
        m = 78
        G = nx.karate_club_graph()
        # G=nx.read_edgelist(DATA_DIR + DATASET_NAME + ".edges")
        
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

        ## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
        # Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
        #node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

        # Embed
        model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

        model.wv.save_word2vec_format(EMBEDDINGS_DIR + DATASET_NAME + ".emb")
        model.save(MODELS_DIR + DATASET_NAME + ".model")
    elif sys.argv[1] == "test":
        G = nx.karate_club_graph()
        print(G)
        model = Word2Vec.load(MODELS_DIR + DATASET_NAME + ".model")
        # print(model.wv.vectors)
        print(model.wv.vectors.shape)
        X_embedded = TSNE(n_components=2).fit_transform(model.wv.vectors)
        alpha=0.9
        label_map = { l: i for i, l in enumerate(np.unique(X_embedded))}
        

        rng = np.random.RandomState(0)
        colors = rng.rand(333)
        plt.figure(figsize=(10,8))
        plt.scatter(X_embedded[:,0], 
                    X_embedded[:,1], 
                    cmap="jet", alpha=alpha)

        plt.show()