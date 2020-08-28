from preprocessing import preprocessing, remove_least_frequent_disease
from glove_skipgram import load_glove

import numpy as np 
import pandas as pd 
from numpy import asarray

import re
import random

def make_data(frame,embedding_dict):
    lst = frame.Disease.unique().tolist()

    skigram_data = []

    for i in frame.index.unique():

        actual = list(frame.Disease.loc[i].values)

        for j in actual:
            skigram_data.append((i,j,1))

            non_context = np.random.choice(list(set(lst)^set(actual)))

            skigram_data.append((i,non_context,0))

    sampled_data = random.sample(skigram_data,len(skigram_data))

    symptom,disease,label= zip(*sampled_data)

    Label = np.array(label,dtype="int32")

    Symptom = pd.Series(list(symptom))

    Disease = pd.Series(list(disease))

    vocab_dict={}

    for i,j in enumerate(Symptom.append(Disease).unique()):
        vocab_dict[j] = i

    Symptom = np.array(Symptom.map(vocab_dict))
    Disease = np.array(Disease.map(vocab_dict))
    
    vocab_size = len(vocab_dict)
    embed_size = 50

    embedding_matrix = np.zeros((vocab_size,embed_size))

    vector_list = []
    for words,index in vocab_dict.items():
    
        for i in words.split():
            if i in embedding_dict.keys():
                vector_list.append(embedding_dict[i])
            arr = np.array(vector_list)
            #print(arr)
            arrsum = arr.sum(axis=0)
            arrsum = arrsum/np.sqrt((arrsum**2).sum()) 
            #print(arrsum)
            embedding_matrix[index,:]=arrsum


    return Symptom,Disease,Label,embedding_matrix,vocab_dict



if __name__ == "__main__":
    frame = preprocessing()
    final_frame = remove_least_frequent_disease(frame)
    embedding_dict= load_glove()
    Symptom,Disease,Label,embedding_matrix,vocab_dict = make_data(final_frame,embedding_dict)
    '''
    print(Symptom)
    print(Disease)
    print(Label)
    #print(embedding_matrix)
    print(Label.shape)
    print(Label.sum())
    '''
    print(vocab_dict)
    