import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import time
from statistics import mean
import tensorflow as tf
import numpy as np
import cv2

test_embavg=np.load("emb.npy")
x2=np.load("lab.npy")

class cosine:
    def similarity(embd):
        l=[]
        for e2 in test_embavg:
            e2=e2.reshape(1,512)
            l.append(cosine_similarity(embd,e2))
        
        #print(l)
        name = x2[np.argmax(np.array(l))]
        print(name)
        return name,max(l)



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

l2=recog.recognize(cv2.imread("/home/soumallyab/Data/EX1057/face-10-2019-04-10 16:07:30.jpg"))

'''
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
    x2.append(j)'''


name,prob=cosine.similarity(l2)

print("Name recognised " + name)
print("Probability ", prob)


