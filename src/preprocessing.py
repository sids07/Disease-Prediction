import pandas as pd 
import re

from collections import Counter

def preprocessing():

    data = pd.read_csv("../input/columbia_main.csv") 
    data = data.iloc[:,1:].drop('Count of Disease Occurrence',axis=1)

    #changing words like "pain/swelling" to "pain swelling"
    data.Symptom = data.Symptom.map(lambda x: re.sub('(.*)\/(.*)',r'\1 \2', str(x)))
    data.Disease = data.Disease.map(lambda x: re.sub('(.*)\/(.*)',r'\1 \2', x))

    #getting rid of paranthesis
    data.Symptom = data.Symptom.map(lambda x: re.sub('(.*)\(.*\)(.*)', r'\1\2',str(x)))
    data.Disease = data.Disease.map(lambda x: re.sub('(.*)\(.*\)(.*)', r'\1\2', x))

    # gets rid of apostrophes and tokens of the sort '\xa0'
    data.Symptom = data.Symptom.map(lambda x: re.sub('\'', '', str(x)))
    data.Disease = data.Disease.map(lambda x: re.sub('\'', '', x))

    data.Symptom = data.Symptom.map(lambda x: re.sub('\\xa0', ' ', str(x)))
    data.Disease = data.Disease.map(lambda x: re.sub('\\xa0', ' ', x))

    #making symptoms as the independent variable

    frame = pd.DataFrame(data.groupby(['Symptom','Disease']).size()).drop(0,axis=1)
    frame = frame.reset_index().set_index('Symptom')

    return frame

def remove_least_frequent_disease(frame):
    count = Counter(frame.index)

    Symptom,Count = list(zip(*sorted(count.items(),key=lambda x: x[1],reverse=True)))

    symptom_count = pd.DataFrame({"Symptoms":Symptom,"No. of Occurrence":Count}).reset_index(drop=True).set_index("Symptoms")

    symptom_count.to_csv("../input/disease_count.csv")

    for i in frame.index.unique():
        if count[i]<6:
            frame.drop(i,inplace=True)

    return frame

if __name__ == "__main__":
    frame = preprocessing()
    print(frame.shape)
    final_frame = remove_least_frequent_disease(frame)
    final_frame.reset_index().to_csv("../input/new.csv")
    print(final_frame)
    print(final_frame.shape)