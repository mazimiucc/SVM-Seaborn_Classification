import os
import numpy as np
import pickle
from sklearn import datasets, svm, metrics
import random
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
with open('ALLVGG.pkl', 'rb') as f:
    dataset_raw = pickle.load(f)

random.shuffle(dataset_raw)

embedding_list_train = list()
label_list_train = list()

embedding_list_test = list()
label_list_test = list()

for emb, label in dataset_raw[:850]:
    #print(emb.shape)
    embedding_list_train.append(emb)
    label_list_train.append(label.split('_')[1])
   

for emb, label in dataset_raw[850:]:
    embedding_list_test.append(emb)
    label_list_test.append(label.split('_')[1])
       
'''    
print('length of embedding train list: {}'.format(len(embedding_list_train)))
print('lenght of label train list: {}'.format(len(label_list_train)))
print('length of embedding test list: {}'.format(len(embedding_list_test)))
print('lenght of label test list: {}'.format(len(label_list_test)))

classifier = svm.SVC(gamma=1, kernel='rbf', C=40)
classifier.fit(embedding_list_train, label_list_train)

expected = label_list_test
predicted = classifier.predict(embedding_list_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
'''
classifier = svm.SVC(gamma=1, kernel='rbf', C=40)
classifier.fit(embedding_list_train, label_list_train)
expected = label_list_test
predicted = classifier.predict(embedding_list_test)

class_names = ["FLRoom", "FMRoom","FOffice","SLRoom","SMRoom","SOffice","Lobby"]
metrics.classification_report(expected, predicted)
print(metrics.classification_report(expected, predicted))
df = pd.DataFrame(metrics.confusion_matrix(expected, predicted), index=class_names, columns=class_names)
print(df)
figsize = (10,7)
fontsize=14
fig = plt.figure(figsize=figsize)
heatmap = sns.heatmap(df, annot=True, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
#plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix: Multi-Class Classification')
fig.savefig('save_as_a_png.png')
