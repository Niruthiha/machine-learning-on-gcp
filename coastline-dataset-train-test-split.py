import pandas as pd

labels= pd.read_csv('labels.csv',names=['gsurl','class'])
labeled_images =pd.read_csv('labeled_images.csv') #training
dict_explanation =pd.read_csv('dict_explanation.csv')

from sklearn.model_selection import train_test_split
#Set features and target data
X=labels.loc[:, labels.columns != 'class']
y=labels['class']

# Split the data into 40% test and 60% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
y_train = pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
labels_train_set = pd.concat([X_train,y_train],axis=1,ignore_index=True)
labels_train_set.columns=['gsurl','class']
labels_test_set = pd.concat([X_test,y_test],axis=1,ignore_index=True)
labels_test_set.columns=['gsurl','class']
sample_test_set = labels_test_set.iloc[100,:]
sample_test_set.to_csv('sample_test_set.csv',index=False)
labels_train_set.to_csv('labels_train_set.csv',index=False)
labels_test_set.to_csv('labels_test_set.csv',index=False)