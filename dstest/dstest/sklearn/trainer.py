from sklearn import svm
from sklearn import datasets
import pickle
import os

#python -m dstest.sklearn.trainer

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
# X: array([[5.1, 3.5, 1.4, 0.2],..., [5.9, 3. , 5.1, 1.8]])
# y: array([0,...,2])
clf.fit(X, y)

model_path = './model/sklearn'
if not os.path.exists(model_path):
    os.makedirs(model_path)

with open("./model/sklearn/model.pkl", "wb") as fp:
    pickle.dump(clf, fp)