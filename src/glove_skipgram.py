from preprocessing import preprocessing, remove_least_frequent_disease

import numpy as np 
import pandas as pd 
from numpy import asarray

import re

def load_glove():
    embedding_dict={}

    with open("../input/glove/glove.6B.50d.txt") as f:
        for line in f:
            values = line.split()
            words = values[0]
            vector = asarray(values[1:],dtype='float32')
            embedding_dict[words]=vector
        f.close()

    print("Loaded {} word vectors".format(len(embedding_dict)))
    return embedding_dict

def get_remove_count(x,embedding_dict):
    count_remove={}
    for i in x.split():
        if i not in embedding_dict.keys():
            count_remove[i] = count_remove.get(i,0)+1
    return count_remove

def get_outofvocabwords(frame,func,embedding_dict):
    disease_out = frame.Disease.map(lambda x: func(x,embedding_dict))
    disease_out_frame = pd.DataFrame(disease_out)
    disease_out_frame.to_csv("../input/disease_out.csv")
    symptom_out = frame.index.map(lambda x: func(x,embedding_dict))
    symptom_out_frame = pd.DataFrame(symptom_out)
    symptom_out_frame.to_csv("../input/symptom_out.csv")

if __name__ == "__main__":
    frame = preprocessing()
    final_frame = remove_least_frequent_disease(frame)
    embedding_dict= load_glove()
    get_outofvocabwords(final_frame,get_remove_count,embedding_dict)
