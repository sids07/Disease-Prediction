from preprocessing import preprocessing, remove_least_frequent_disease
from glove_skipgram import load_glove
from data import make_data

import numpy as np 
import pandas as pd 
from numpy import asarray

import re
import random

from tensorflow.keras.layers import Concatenate, Dot, Dense, Reshape, Dense, Reshape,Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
import tensorflow


'''
class skipgram(layers.Layer):
    def __init__(self,vocab_size,embed_size,embedding_matrix,input_size,output_size):
        super(skipgram,self).__init()

        self.embedding = layers.Embedding(input_dim=vocab_size,
                     output_dim=vector_size,
                     input_length=1,
                     name='embedding',
                     trainable=True,
                     weights = [embedding_matrix])
        self.reshape_1 = layers.Reshape((embed_size,1))

        self.dot = layers.Dot(axes=1)
        self.reshape_2 = layers.Reshape((1,))
        self.dense = layers.Dense(output_size,activation='sigmoid')

    def call(self,input_target,input_target):

        target = self.embedding(input_target)
        target = self.reshape_1(target)

        context = self.embedding(input_context)
        context = self.reshape_1(context)

        merged = self.dot([context,target])
        merged = self.reshape_2(merged)

        output = self.dense(merged)

        return output
'''
def cosine_similarity(x, y):
    
    # Compute the dot product between x and y 
    dot = np.dot(x,y)
    # Compute the L2 norm of x 
    norm_x = np.sqrt(np.sum(x**2))
    # Compute the L2 norm of y
    norm_y = np.sqrt(np.sum(y**2))
    # Compute the cosine similarity
    cosine_similarity = dot/(norm_x * norm_y)
    return cosine_similarity

def get_keys(val):
    for (keys,value) in vocab_dict.items():
        if val == value:
            return keys


if __name__ == "__main__":
    frame = preprocessing()
    final_frame = remove_least_frequent_disease(frame)
    embedding_dict= load_glove()
    Symptom,Disease,Label,embedding_matrix,vocab_dict = make_data(final_frame,embedding_dict)

    vocab_size = len(vocab_dict)
    vector_size = 50

    input_target = Input(shape=(1,))
    input_context = Input(shape=(1,))
    
    embedding = Embedding(input_dim=vocab_size,
                     output_dim=vector_size,
                     input_length=1,
                     name='embedding',
                     trainable=True)
    embedding.build((None,))
    embedding.set_weights([embedding_matrix])

    context = embedding(input_context)
    context = Reshape((vector_size,1))(context)

    target = embedding(input_target)
    target = Reshape((vector_size,1))(target)

    dot_1 = Dot(axes=1)([context,target])
    dot_1 = Reshape((1,))(dot_1)
    out_1 = Dense(1,activation='sigmoid')(dot_1)

    model = Model(inputs=[input_context,input_target],outputs=out_1)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(x=[Symptom,Disease],y=Label,epochs=30)

    new_vecs = model.layers[2].get_weights()[0]

    tensorflow.keras.utils.plot_model(model, to_file='model_combined.png')
    np.save('../input/similarity.npy', new_vecs)

    similarity_score = float(0.6)

    # enter the symptom
    symp = input('Enter symptom for which similar symptoms are to be found: ')
    print ('\nThe similar symptoms are: ')

    # loop through the symptoms in the data set and find the symptoms with cosine similarity greater than 'similarity_score'
    for i in set(tuple(Symptom.tolist())):
        if float(cosine_similarity(new_vecs[i], new_vecs[vocab_dict[symp]])) > similarity_score:
        # remove the same symptom from the list of outputs
            similar_symp = get_keys(i)

            if similar_symp != symp:
                print (similar_symp)