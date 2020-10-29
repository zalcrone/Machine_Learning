import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train  = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")
y_testing = pd.read_csv("gender_submission.csv")



x_train = dataset_train.iloc[:,[2,4,5,6,7,9,11]].values
y_train = dataset_train.iloc[:, 1].values
x_test = dataset_test.iloc[:,[1,3,4,5,6,8,10]].values
y_test = y_testing.iloc[:, 1].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x_train[:, 1] = le.fit_transform(x_train[:, 1])
x_test[:, 1] = le.fit_transform(x_test[:, 1])

x_train = pd.DataFrame(x_train)
x_train = x_train.apply(lambda x: x.fillna(x.value_counts().index[0]))
x_test = pd.DataFrame(x_test)
x_test = x_test.apply(lambda x: x.fillna(x.value_counts().index[0]))



# =============================================================================
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x_train[:, 0:-1])
# x_train[:, 0:-1] = imputer.transform(x_train[:, 0:-1])
# imputer.fit(x_test[:, 0:-1])
# x_test[:, 0:-1] = imputer.transform(x_test[:, 0:-1])
# =============================================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [6])], remainder = 'passthrough')
x_train = np.array(ct.fit_transform(x_train))
x_test = np.array(ct.fit_transform(x_test))


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))  
