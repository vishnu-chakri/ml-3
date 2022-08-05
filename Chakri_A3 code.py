#Import libraies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import friedmanchisquare
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  scale
import time
import os

#Explore data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "Chakri_A3 dataset.csv"))
print(data.shape)


# removing the null values
data.isna().sum()


#correlation heat map for all the attributes
plt.figure(figsize=(15,8))
corr = data.corr()
sns.heatmap(corr, annot= True, annot_kws={"size":6})
plt.show()




corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']


# selecting the attributes which have corelation grater than 0.1
new_data = data[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]


x = new_data
y = data['DEATH_EVENT']

#linear regression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr=LogisticRegression(max_iter=10000)
lr.fit(x_train,y_train)
p1=lr.predict(x_test)
s1=accuracy_score(y_test,p1)
f1=f1_score(y_test,p1)
print("Linear Regression Success Rate for accuracy:", "{:.2f}%".format(100*s1))
print("Linear Regression Success Rate for f1-score:", "{:.2f}%".format(100*f1))

# Gradient booser classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
p2=gbc.predict(x_test)
s2=accuracy_score(y_test,p2)
f2=f1_score(y_test,p2)
print("Gradient Booster Classifier Success Rate for accuracy:", "{:.2f}%".format(100*s2))
print("Gradient Booster Classifier Success Rate for f1_score:", "{:.2f}%".format(100*f2))

#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
p3=rfc.predict(x_test)
s3=accuracy_score(y_test,p3)
f3=f1_score(y_test,p3)
print("Random Forest Classifier Success Rate for accuracy:", "{:.2f}%".format(100*s3))
print("Random Forest Classifier Success Rate for f1-score:", "{:.2f}%".format(100*f3))


#svm
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
p4=svm.predict(x_test)
s4=accuracy_score(y_test,p4)
f4=f1_score(y_test,p4)
(y_test,p4)
print("Support Vector Machine Success Rate for accuracy:", "{:.2f}%".format(100*s4))
print("Support Vector Machine Success Rate for f1-score:", "{:.2f}%".format(100*f4))


#K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
scorelist=[]
scorelist2 = []
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    p5=knn.predict(x_test)
    s5=accuracy_score(y_test,p5)
    f5=f1_score(y_test,p5)
    scorelist.append(round(100*s5, 2))
    scorelist2.append(round(100*f5, 2))

print("K Nearest Neighbors Top 5 Success Rates for accuracy:")
print(sorted(scorelist,reverse=True)[:5])

print("K Nearest Neighbors Top 5 Success Rates for f1-score:")
print(sorted(scorelist2,reverse=True)[:5])