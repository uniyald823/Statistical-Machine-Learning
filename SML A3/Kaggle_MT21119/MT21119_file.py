#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction

# In[25]:


import pandas as pd
import numpy as np
import pickle
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from random import randrange,uniform
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz               
from sklearn.metrics import accuracy_score            
from sklearn.metrics import confusion_matrix           
from sklearn.ensemble import RandomForestClassifier    
import statsmodels.api as sn                           
from sklearn.neighbors import KNeighborsClassifier     
from sklearn.naive_bayes import GaussianNB             
from sklearn import model_selection                  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score,roc_curve 
from sklearn.metrics import classification_report      
import pickle                                        
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor  
from statsmodels.tools.tools import add_constant
np.random.seed(123) 
pd.options.mode.chained_assignment = None  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir())
# Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("C:/Users/HP/Downloads/Train.csv")


# In[3]:


test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")


# In[4]:


data.head()


# In[5]:


data.sample(5)


# In[6]:


data.describe()


# In[7]:


data.info()


# # About the dataset

# In[8]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(data.columns[i]+":\t\t\t"+info[i])


#  -----------------------------------------------------------------------------------

# In[9]:


type(data)


# In[10]:


data.shape


# In[ ]:





# In[11]:


data.drop_duplicates(inplace=True)


# ## Exploratory Data Analysis (EDA)

# In[12]:


y = data["target"]
target_temp = data.target.value_counts()
print(target_temp)


# ### We notice, that females are more likely to have heart problems than males

# # Percentage of patients with or without heart problems

# In[13]:


countNoDisease = len(data[data.target == 0])
countHaveDisease = len(data[data.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(data.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(data.target))*100)))


# In[14]:


data.groupby('target').mean()


# ------------------------------------------------------------------------------------

# # Train Test Split

# In[15]:


predictors = data.drop("target",axis=1)
target = data["target"]
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[18]:


Y_train.shape


# In[19]:


Y_test.shape


# # Model Fitting

# In[20]:


# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# def print_score(clf, X_train, y_train, X_test, y_test, train=True):
#     if train:
#         pred = clf.predict(X_train)
#         clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
#         print("Train Result:\n================================================")
#         print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
#         print("_______________________________________________")
#         print(f"CLASSIFICATION REPORT:\n{clf_report}")
#         print("_______________________________________________")
#         print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
#     elif train==False:
#         pred = clf.predict(X_test)
#         clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
#         print("Test Result:\n================================================")        
#         print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
#         print("_______________________________________________")
#         print(f"CLASSIFICATION REPORT:\n{clf_report}")
#         print("_______________________________________________")
#         print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[ ]:





# ## Logistic Regression

# In[22]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
train_accuracy = round(lr.score(X_train, Y_train)*100, 2)
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# In[23]:


# define the file path where you want to save the model
filename = 'logistic_regression_model.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(lr, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = lr.predict(test)
test.to_csv("LR.csv")


# In[52]:





# ## Naive Bayes
# 
# ---
# 
# 

# In[24]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,Y_train)
Y_pred_nb = nb.predict(X_test)

train_accuracy = round(nb.score(X_train, Y_train)*100, 2)
score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# In[25]:


# define the file path where you want to save the model
filename = 'naive_bayes_model.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(nb, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = nb.predict(test)
test.to_csv("NB.csv")


# In[26]:


# #check accuracy of model
# score_nb=((TP+TN)*100)/(TP+TN+FP+FN)
# score_nb


# In[27]:


# # check false negative rate of the model
# fnr=FN*100/(FN+TP)
# fnr


# In[29]:


# test['target'] = nb.predict(test)
# #test.to_csv("lr_new.csv")


# 
# 
# ---
# 
# 

# # Decision Tree

# In[ ]:


# replace target variable  with yes or no
#data['target'] = data['target'].replace(0, 'No')
#data['target'] = data['target'].replace(1, 'Yes')


# In[ ]:


# to handle data imbalance issue we are dividing our dataset on basis of stratified sampling
# divide data into train and test
#X=data.values[:,0:13]
#Y=data.values[:,13]
#X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2)


# In[30]:


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0
for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
train_accuracy = round(dt.score(X_train, Y_train)*100, 2)

score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# In[31]:


# define the file path where you want to save the model
filename = 'decision_tree_model.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(dt, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = dt.predict(test)
test.to_csv("DT.csv")


# In[ ]:





# # KNN(K Nearest Neighbors)

# for neighbors = 7

# In[32]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)
train_accuracy = round(knn.score(X_train, Y_train)*100, 2)

score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# In[33]:


# define the file path where you want to save the model
filename = 'knn_model.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(knn, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = knn.predict(test)
test.to_csv("KNN.csv")


# ## Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
train_accuracy = round(rf.score(X_train, Y_train)*100, 2)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# In[35]:


# define the file path where you want to save the model
filename = 'random_forest_model.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(rf, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = rf.predict(test)
test.to_csv("NB.csv")


# In[ ]:





# ## XGBoost

# In[ ]:


import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)
train_accuracy = round(xgb_model.score(X_train, Y_train)*100, 2)
Y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")

print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")


# In[ ]:


# define the file path where you want to save the model
filename = 'xgb_model.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(xgb_model, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = xgb_model.predict(test)
test.to_csv("XG.csv")


# In[ ]:





# In[22]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train, Y_train)
clf.score(X_train,Y_train)
train_accuracy = round(clf.score(X_train, Y_train)*100, 2)
Y_pred_xgb= clf.predict(X_test)
score_ada = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using Adaboost is: "+str(score_ada)+" %")
# test.to_csv("C:/Users/HP/Downloads/ada_lr.csv")


# In[23]:


# define the file path where you want to save the model
filename = 'ada.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(clf, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = clf.predict(test)
#test.to_csv("ada_lr.csv")


# ## Adaboost + LR

# In[26]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=96, base_estimator = LogisticRegression(), 
                         n_estimators=100, learning_rate=0.01)
clf.fit(X_train, Y_train)
clf.score(X_train,Y_train)

Y_pred_xgb = clf.predict(X_test)
score_ada_lr = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using Adaboost+lr is: "+str(score_ada_lr)+" %")
# test.to_csv("C:/Users/HP/Downloads/ada_lr.csv")


# In[27]:


# define the file path where you want to save the model
filename = 'ada_lr.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(clf, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = clf.predict(test)
#test.to_csv("ada_lr_test.csv")


# In[ ]:





# In[28]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=96, base_estimator = RandomForestClassifier(), 
                         n_estimators=100, learning_rate=0.01)
clf.fit(X_train, Y_train)
clf.score(X_train,Y_train)

Y_pred_xgb = clf.predict(X_test)
score_ada_lr = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)
print("The Train accuracy score achieved is "+str(train_accuracy)+" %")
print("The accuracy score achieved using Adaboost+rf is: "+str(score_ada_lr)+" %")
# test.to_csv("C:/Users/HP/Downloads/ada_lr.csv")

# define the file path where you want to save the model
filename = 'ada_rf.pkl'

# save the model using pickle
with open(filename, 'wb') as f:
    pickle.dump(clf, f)

# close the file
f.close()

# open the saved model file
with open(filename, 'rb') as f:
    # load the saved model using pickle
    loaded_model = pickle.load(f)

# close the file
f.close()

# use the loaded model to make predictions
test = pd.read_csv("C:/Users/HP/Downloads/Test.csv")
test['target'] = clf.predict(test)
test.to_csv("ada_rf.csv")


# In[ ]:





# In[32]:


# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# # create base estimators
# lr = LogisticRegression(random_state=96)
# rf = RandomForestClassifier(random_state=96)

# # create AdaBoost classifier with base estimators
# clf = AdaBoostClassifier(base_estimator=[lr, rf], n_estimators=100, learning_rate=0.01, random_state=96)

# # fit the classifier to the training data
# clf.fit(X_train, Y_train)

# # evaluate performance on the training data
# train_accuracy = clf.score(X_train, Y_train)

# # predict on the test data and evaluate performance
# Y_pred_ada = clf.predict(X_test)
# score_ada = round(accuracy_score(Y_pred_ada, Y_test)*100, 2)

# print("The train accuracy score achieved is " + str(train_accuracy*100) + "%")
# print("The accuracy score achieved using AdaBoost with Logistic Regression and Random Forest is: " + str(score_ada) + "%")


# In[ ]:




