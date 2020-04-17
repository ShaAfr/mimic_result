#code to finding distance of the negative samples from the positive samples

import pandas as pd
from itertools import chain

#take input
train = pd.read_csv('dump/train_listfile.csv')

#true value subdataframe finding for KNN training and find index of the true labels so that we can match it with dump folder
true_train = train[train['y_true']==1]
true_train_label = true_train.index.to_list()
print('number of true labels: ', len(true_train_label))

full_array = []
#dump folder accessing with the index of true labels [change the range to...]
for f in range(len(true_train_label)):
    f_num = true_train_label[f]
    filename = 'dump/train_'+ str(f_num) + '.csv'
    filein = pd.read_csv(filename)

    #flaten the dataframe (48X76) to 2d list to 1d list
    df_to_2dlist = filein.values.tolist()
    twoD_to_1dlist = list(chain.from_iterable(df_to_2dlist))
    full_array.append(twoD_to_1dlist)

print('number of true labels considered: ', len(full_array))



#apply KNN
from sklearn.neighbors import NearestNeighbors
import numpy as np

samples = full_array
neigh = NearestNeighbors(n_neighbors = 3)
neigh.fit(samples)
 

#test on the label 0 [remove second line later]
false_train = train[train['y_true']==0]
###false_train = false_train.head()
false_train_label = false_train.index.to_list()
print("number of false labels", len(false_train_label))

full_array = []
#dump folder accessing with the index of false labels
for f in range(len(false_train_label)):
     f_num = false_train_label[f]
     filename = 'dump/train_'+ str(f_num) + '.csv'
     filein = pd.read_csv(filename) 
     #flaten the dataframe (48X76) to 2d list to 1d list
     df_to_2dlist = filein.values.tolist()
     twoD_to_1dlist = list(chain.from_iterable(df_to_2dlist))
     full_array.append(twoD_to_1dlist) 

print('number of false labels', len(full_array)) 


#KNN model: input the test data
dist, nodes = neigh.kneighbors(full_array)
distances = np.mean(dist,axis=1)
distances = distances.tolist()
distances

false_train['knn_dist'] = distances

false_train.to_csv('dist_train_false_labels.csv', index = False)




 


 










  

