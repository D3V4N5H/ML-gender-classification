# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
# CHALLENGE - create 3 more classifiers...

# So as per http://scikit-learn.org/stable/tutorial/machine_learning_map/
# I used Linear SVC first
# 1 SVC
c1=svm.SVC()
c1.fit(X,Y)
# Then on that webpage was KNeighbors Classifier
# 2 Nearest Neighbors
n_neighbors=11
c2 = neighbors.KNeighborsClassifier(n_neighbors)
c2.fit(X,Y)
# 3 Ensemble Classifiers
c3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X,Y)


treeprediction = clf.predict([[190, 70, 43]])
svmprediction = c1.predict([[190, 70, 43]])[0]
knprediction = c2.predict([[190, 70, 43]])[0]
ensembleprediction = c3.predict([[190, 70, 43]])[0]

# CHALLENGE compare their results and print the best one!

print("Tree:",treeprediction[0])
print("SVM:",svmprediction)
print("neighbors: ",knprediction)
print("ensemble: ",ensembleprediction)