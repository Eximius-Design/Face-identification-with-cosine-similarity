import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import time
from statistics import mean
import tensorflow as tf
import numpy as np
import cv2

class Recognition:
    
    def __init__(self, model_path):
        
        graph = tf.Graph()
        with graph.as_default():
            with open(model_path, 'rb') as f:
                graph_def = tf.GraphDef.FromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.graph = graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        print("Recognition Model Graph Initialized")

    def recognize(self, face):

        face = cv2.resize(face,(160,160))
        face = face.reshape(1,160,160,3)
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        feed_dict = { images_placeholder:face, phase_train_placeholder:False }
        
        embeddings = self.sess.run(embeddings, feed_dict=feed_dict)
        return embeddings

recog=Recognition("/home/soumallyab/Data/Recognition_facenet.pb")

test_emb2=[]
x2=[]
for j in ['EX0870','EX1027-2','EX1057','EX1093-2','EXC169','EX0314-2']:
    for i in os.listdir("/home/soumallyab/Data/"+j):
        test_emb2.append(recog.recognize(cv2.imread("/home/soumallyab/Data/"+j+"/"+i))) #embeddings
	
p=0
test_embavg=[]
for j in ['EX0870','EX1027-2','EX1057','EX1093-2','EXC169','EX0314-2']:
    test=[]
    for i in range(len(os.listdir("/home/soumallyab/Data/"+j))):
        test.append(test_emb2[p])
        p=p+1
    test_embavg.append(np.mean(np.array(test),axis=0))
    x2.append(j)

test_embavg = np.array(test_embavg)
x2=np.array(x2)

np.save("emb.npy",test_embavg)
np.save("lab.npy",x2)


