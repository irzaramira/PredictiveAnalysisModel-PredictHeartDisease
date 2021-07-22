# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:08:18 2021

@author: irzar
"""

# -- import data --
import pandas as pd

data = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
#------------------

# -- EDA --
data.head()
data.info()
data.describe()

data.target.value_counts()

import seaborn as sns
import matplotlib.pyplot as plt 

# deskripsi dan plot target

sns.countplot(x = "target", data = data, palette = ['black', 'grey'])
plt.show()


sumnotsick = len(data[data.target == 0])
sumsick = len(data[data.target == 1])
print('')
print("Percentage of Patients who have Heart Disease= {:.2f}%".
      format((sumsick / (len(data.target))* 100 )))

print("Percentage of Patients without Heart Disease = {:.2f}%".
      format((sumnotsick / (len(data.target))* 100 )))

#describe sex
sns.countplot(x='sex', data=data , palette = ['black', 'grey'])
plt.xlabel ("Attribute for sex, 0 = Women, 1 = Men")
plt.show()


sumwomen = len(data[data.sex == 0])
summen = len(data[data.sex == 1])
print('')
print("Presentase Pasien Perempuan = {:.2f}%".
      format((sumwomen / (len(data.sex))* 100 )))

print("Presentase Pasien Laki-Laki = {:.2f}%".
      format((summen / (len(data.sex))* 100 )))
# ---------

# -- pre processing data -- 
y = data.target.values
x = data.drop(['target'],axis=1)

# dummy data to help increase accuracy
dummy_slope = pd.get_dummies(data['slope'], prefix = "slope")
frames = [data, dummy_slope]
data = pd.concat(frames, axis = 1)
data = data.drop(columns = ['slope'])

# normalize data
import numpy as np
x_normalize = (x - np.min(x)) / (np.max(x) - np.min(x)).values

# splitting data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

# transpose
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
# -------------------------

# --- MACHINE LEARNING MODEL ---
acctotal = {}

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
knn.predict(x_test.T)

print("KNN Score: {:.2f}%".format(1, knn.score(x_test.T, y_test.T)*100))

# find the best value of n
results = []
for i in range(1,20):
    knn_exp = KNeighborsClassifier(n_neighbors = i) #n_neighbors berarti k = 2
    knn_exp.fit(x_train.T, y_train.T)
    results.append(knn_exp.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), results)
plt.xticks(np.arange(1,20,1))
plt.xlabel("Nilai K")
plt.ylabel("Skor")
plt.show()

accKNN = max(results)*100
acctotal['KNN'] = accKNN
print("Maximum KNN Score is {:.2f}%".format(accKNN))

# SVM
from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(x_train.T, y_train.T)

accSVM = svm.score(x_test.T, y_test.T)*100
acctotal['SVM'] = accSVM
print("\nTesting Accuracy SVM : {:.2f}%".format(accSVM))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
naivebayes = GaussianNB()
naivebayes.fit(x_train.T, y_train.T)

accNaiveBayes = naivebayes.score(x_test.T, y_test.T)*100
acctotal['NaiveBayes'] = accNaiveBayes
print("\nTesting Accuracy Naive Bayes : {:.2f}%".format(accNaiveBayes))

#Decision Tree Algoritma
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train.T, y_train.T)

accdecisiontree = decisiontree.score(x_test.T, y_test.T)*100
acctotal['DecisionTree'] = accdecisiontree
print("\nTesting Accuracy Decision Tree : {:.2f}%".format(accdecisiontree))

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 1000, random_state = 1)
randomforest.fit(x_train.T, y_train.T)

accrandomforest = randomforest.score(x_test.T, y_test.T)*100
acctotal['Random Forrest'] = accrandomforest
print("\nTesting Accuracy Random Forrest : {:.2f}%".format(accrandomforest))

# Results and Evaluation
from sklearn.metrics import confusion_matrix, classification_report

#KNN EVALUATION
knn_conf = KNeighborsClassifier(n_neighbors = 1)
knn_conf.fit(x_train.T, y_train.T)
y_head_knn_train = knn_conf.predict(x_train.T)
y_head_knn_test = knn_conf.predict(x_test.T)

confusionmatrix_knn_train = confusion_matrix(y_train, y_head_knn_train)
print(confusionmatrix_knn_train)
clf_report_knn_train = pd.DataFrame(classification_report(y_train.T,y_head_knn_train, output_dict = True))
print("KNN CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_knn_train)
print("_______________________________________________________________________")

confusionmatrix_knn_test = confusion_matrix(y_test, y_head_knn_test)
print(confusionmatrix_knn_test)
clf_report_knn_test = pd.DataFrame(classification_report(y_test.T,y_head_knn_test, output_dict = True))
print("KNN CLASSIFICATION REPORT TEST DATA:\n",clf_report_knn_test)
print("_______________________________________________________________________")


#SVM EVALUATION
y_head_svm_train = svm.predict(x_train.T)
y_head_svm_test = svm.predict(x_test.T)

confusionmatrix_svm_train = confusion_matrix(y_train, y_head_svm_train)
print(confusionmatrix_svm_train)
clf_report_svm_train = pd.DataFrame(classification_report(y_train.T,y_head_svm_train, output_dict = True))
print("SVM CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_svm_train)
print("_______________________________________________________________________")

confusionmatrix_svm_test = confusion_matrix(y_test, y_head_svm_test)
print(confusionmatrix_svm_test)
clf_report_svm_test = pd.DataFrame(classification_report(y_test.T,y_head_svm_test, output_dict = True))
print("SVM CLASSIFICATION REPORT TEST DATA:\n",clf_report_svm_test)
print("_______________________________________________________________________")

#NaiveBayes EVALUATION
y_head_naivebayes_train = naivebayes.predict(x_train.T)
y_head_naivebayes_test = naivebayes.predict(x_test.T)

confusionmatrix_naivebayes_train = confusion_matrix(y_train, y_head_naivebayes_train)
print(confusionmatrix_naivebayes_train)
clf_report_naivebayes_train = pd.DataFrame(classification_report(y_train.T,y_head_naivebayes_train, output_dict = True))
print("NaiveBayes CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_naivebayes_train)
print("_______________________________________________________________________")

confusionmatrix_naivebayes_test = confusion_matrix(y_test, y_head_naivebayes_test)
print(confusionmatrix_naivebayes_test)
clf_report_naivebayes_test = pd.DataFrame(classification_report(y_test.T,y_head_naivebayes_test, output_dict = True))
print("NaiveBayes CLASSIFICATION REPORT TEST DATA:\n",clf_report_naivebayes_test)
print("_______________________________________________________________________")


#Decision Tree EVALUATION
y_head_decisiontree_train = decisiontree.predict(x_train.T)
y_head_decisiontree_test = decisiontree.predict(x_test.T)

confusionmatrix_decisiontree_train = confusion_matrix(y_train, y_head_decisiontree_train)
print(confusionmatrix_decisiontree_train)
clf_report_decisiontree_train = pd.DataFrame(classification_report(y_train.T,y_head_decisiontree_train, output_dict = True))
print("Decision Tree CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_decisiontree_train)
print("_______________________________________________________________________")

confusionmatrix_decisiontree_test = confusion_matrix(y_test, y_head_decisiontree_test)
print(confusionmatrix_decisiontree_test)
clf_report_decisiontree_test = pd.DataFrame(classification_report(y_test.T,y_head_decisiontree_test, output_dict = True))
print("Decision Tree CLASSIFICATION REPORT TEST DATA:\n",clf_report_decisiontree_test)
print("_______________________________________________________________________")

#Random Forest EVALUATION
y_head_randomforest_train = randomforest.predict(x_train.T)
y_head_randomforest_test = randomforest.predict(x_test.T)

confusionmatrix_randomforest_train = confusion_matrix(y_train, y_head_randomforest_train)
print(confusionmatrix_randomforest_train)
clf_report_randomforest_train = pd.DataFrame(classification_report(y_train.T,y_head_randomforest_train, output_dict = True))
print("Random Forest CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_randomforest_train)
print("_______________________________________________________________________")

confusionmatrix_randomforest_test = confusion_matrix(y_test, y_head_randomforest_test)
print(confusionmatrix_randomforest_test)
clf_report_randomforest_train = pd.DataFrame(classification_report(y_test.T,y_head_randomforest_test, output_dict = True))
print("Random Forest CLASSIFICATION REPORT Test DATA:\n",clf_report_randomforest_train)
print("_______________________________________________________________________")