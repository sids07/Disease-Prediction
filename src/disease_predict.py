from preprocessing import preprocessing, remove_least_frequent_disease

import numpy as np 
import pandas as pd 

import re

import pickle

from sklearn.naive_bayes import MultinomialNB

def make_dataset(frame):
    frame = frame.reset_index()

    data = pd.get_dummies(frame,columns=['Symptom'],prefix='',prefix_sep='')
    data = data.groupby(data.Disease).sum().reset_index()
    
    X = data.iloc[:,1:]
    y = data.Disease
    return X,y
    
def predict(sample,x):
    total_columns = X.columns
    disease_idx={}
    for i,dis in enumerate(total_columns):
        disease_idx[dis] = i

    data = np.zeros((len(disease_idx),))

    for i in sample:
        data[disease_idx[i]] =1
    result=model.predict_proba(data.reshape(1,-1))[0]
    out=dict(zip(model.classes_,result))
    sort_orders = sorted(out.items(), key=lambda x: x[1], reverse=True)
    disease=sort_orders[0:10]
    return disease

if __name__ == "__main__":
    frame = preprocessing()
    final_frame = remove_least_frequent_disease(frame)
    print(final_frame[final_frame['Disease']=='paranoia'])
    X ,y= make_dataset(final_frame)
    X.to_csv("../input/all_x.csv")
    #print(X.head())
    
    model = MultinomialNB()

    model = model.fit(X,y)

    accuracy_score = model.score(X,y)

    disease_pred = model.predict(X)
    disease_real = y.values

    print("accuracy_score : {}".format(accuracy_score))
    

    for i in range(0, len(disease_real)):
        if disease_pred[i]!=disease_real[i]:
            print ('Pred: {0} Actual:{1}'.format(disease_pred[i], disease_real[i]))
    
    #print(X.columns.tolist())

    pickle.dump(model,open('models'+'.p','wb'))
    
    sample = ['agitation','blackout','feeling suicidal','mood depressed','homelessness']
    predictions = predict(sample,X)

    print("Predicted Disease is:{} with percentage:{}".format(predictions[0][0],predictions[0][1]*100))
    
