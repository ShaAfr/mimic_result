import pandas as pd

filetype = 'train'

analysis = pd.read_csv('dump/' + filetype + '_listfile.csv')
print('file shape: ', analysis.shape)

#selecting positive ones
positiveones = analysis[analysis['y_true'] == 1]
print('positive ones shape: ' , positiveones.shape)


#select negative ones
falseLabels = pd.read_csv('dist_' + filetype + '_false_labels_nearmiss3.csv')
print('false label shape', falseLabels.shape)

sorted_ = falseLabels.sort_values(by=['knn_dist'])

print('sorted shape: ', sorted_.shape)

negativeOnes = sorted_[['stay', 'y_true']].head(positiveones.shape[0])

print('size of selected negative samples: ', negativeOnes)

full_list = positiveones.append(negativeOnes)
print(full_list)

full_list.to_csv(filetype + '_nearmiss3.csv', index = False)
