# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:00:45 2021

@author: Joey
"""

# import packages

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
import random 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
LE = LabelEncoder()


mushroom_data = pd.read_csv("C:/Users/Joey/Downloads/archive(2)/mushrooms.csv")

#survey_response['Punctuality'] = LE.fit_transform(survey_response['Punctuality'].tolist())
#survey_response['Lying'] = LE.fit_transform(survey_response['Lying'].tolist())
#survey_response['Internet usage'] = LE.fit_transform(survey_response['Internet usage'].tolist())
#survey_response['Left - right handed'] = LE.fit_transform(survey_response['Left - right handed'].tolist())
#survey_response['Smoking'] = LE.fit_transform(survey_response['Smoking'].tolist())
#survey_response['Alcohol'] = LE.fit_transform(survey_response['Alcohol'].tolist())
#survey_response['Gender'] = LE.fit_transform(survey_response['Gender'].tolist())
#survey_response['Education'] = LE.fit_transform(survey_response['Education'].tolist())
#survey_response['Only child'] = LE.fit_transform(survey_response['Only child'].tolist())
#survey_response['Village - town'] = LE.fit_transform(survey_response['Village - town'].tolist())
#
#
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(survey_response.iloc[16])

#mushroom_coded = pd.get_dummies(mushroom_data.iloc[:,2:])
x = mushroom_data["class"].value_counts().reset_index()
x["new_class"] = mushroom_data["class"].value_counts().reset_index()["index"]
x = x.drop(columns={"index"})
plt.bar(x.new_class, x['class'])
plt.xlabel('Type of Mushroom')
plt.xticks(x.new_class, ['Edible','Poisonous'])
plt.show()
for i in range(1,mushroom_data.shape[1]):
    print(mushroom_data.iloc[:,i].value_counts())
    
# Note that Viel-type is a single value
    
mushroom_data = mushroom_data.drop(columns={"veil-type"})


mushroom_coded = mushroom_data.iloc[:,:].apply(lambda col: LE.fit_transform(col))
plt.figure(figsize = (12,9))
ax = sns.heatmap(mushroom_coded.corr())

# create tables for those with high correlation with class eg bruises, gil-color, ring-type,gil-size


# If it does have bruines, 82% chance that the mushroom is edible according to the data, if it doesn't 31% edible, 69% chance it's poisnous
pd.crosstab(index= mushroom_data['class'], columns = mushroom_data['bruises'])


# Multiple colors are more dominant in edible vs poisonous and vice versa
pd.crosstab(index= mushroom_data['class'], columns = mushroom_data['gill-color'])


# Some ring typs are strictly for one class
pd.crosstab(index= mushroom_data['class'], columns = mushroom_data['ring-type'])

# Narrow gill size seems to indicate poisonous, broad might indicate edible
pd.crosstab(index= mushroom_data['class'], columns = mushroom_data['gill-size'])




x_train, x_test, y_train, y_test = train_test_split(mushroom_coded.iloc[:,1:], mushroom_coded['class'], test_size=.3, random_state  = 2047)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

clf = tree.DecisionTreeClassifier(random_state=2047)
clf = clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))

# Note that the main splits don't include the highly correlated variables, may indicate some colinearlity issues in the data
#plt.figure(figsize=(30,10))
#tree.plot_tree(clf, fontsize = 10, feature_names = mushroom_coded.iloc[:,2:].columns, filled = True, class_names = ["edible","poisonous"])
#plt.show()

# 
max_depth = []
entropy_accuracy_train = []
entropy_accuracy_test = []

for i in range(1,10):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = i, random_state = 2047)
    clf.fit(x_train,y_train)
    entropy_accuracy_train.append(f1_score(y_train, clf.predict(x_train)))
    entropy_accuracy_test.append(f1_score(y_test, clf.predict(x_test)))
    max_depth.append(i)
    
# Achieving a max a
plt.plot(range(1,10), entropy_accuracy_train, color = 'g', label = "Training data")
plt.plot(range(1,10), entropy_accuracy_test, color = 'r', label = "Test data")
plt.title("Mushroom Classifcation: Max Depth v Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

print("BEST DEPTH: " + str(entropy_accuracy_test.index(max(entropy_accuracy_test)) + 1))
clf = tree.DecisionTreeClassifier(max_depth = entropy_accuracy_test.index(max(entropy_accuracy_test)) + 1, random_state = 2047, criterion='entropy')
clf = clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))


train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.01, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    np.random.seed(207888)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("Mushroom Classifcation: CV Decision Tree")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["edible","poisonous"])
plt.show()

########################################################################################
#               BOOSTED DECISION TREE 
########################################################################################
f1_array_train = []
f1_array_test = []
array_tree_values = list(range(1, 100,10))
for tree_ in array_tree_values:
    clf = GradientBoostingClassifier(n_estimators = tree_, random_state = 2047)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))

plt.plot(array_tree_values, f1_array_train, color = 'g', label = "Training data")
plt.plot(array_tree_values, f1_array_test, color = 'r', label = "Test data")
plt.xlabel("Number of Estimators")
plt.ylabel("F1 Score")
plt.title("Mushroom Classifcation: Gradient Boosting DT")
plt.legend()
plt.show()



tree_boost = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = {'max_depth': [1,2,3],'n_estimators': array_tree_values,'learning_rate':np.linspace(.001,1,3)}, cv = 10)

tree_boost.fit(x_train,y_train)

print(tree_boost.best_params_)


# FIND BEST RESULTS
gbc_best = GradientBoostingClassifier(max_depth =tree_boost.best_params_['max_depth'], n_estimators =tree_boost.best_params_['n_estimators'], learning_rate =tree_boost.best_params_['learning_rate'], random_state = 2047)



clf = gbc_best.fit(x_train, y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))


train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.01, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    np.random.seed(207888)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("Mushroom Classifcation: Cross Validation Gradient Boosting DT ")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["edible","poisonous"])
plt.show()

##############################################################################
#                            SVM
#################################################################################

f1_array_train = []
f1_array_test = []
kernel_type = ["linear", "rbf", "poly"]
for i in kernel_type:
    clf = SVC(kernel = i, random_state = 2047)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))


plt.plot(kernel_type, f1_array_train, color = 'g', label = "Training data")
plt.plot(kernel_type, f1_array_test, color = 'r', label = "Test data")
plt.xlabel("Kernel Type")
plt.ylabel("F1 Score")
plt.title("Mushroom Classification: Support Vector Machines")
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["edible","poisonous"])
plt.show()


clf = SVC(random_state = 2047, kernel = 'poly')
clf.fit(x_train,y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))

train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.01, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    np.random.seed(207888)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("Mushroom Classifcation: CV SVM, Poly")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["edible","poisonous"])
plt.title("Mushroom Classifcation: Confusion Matrix")
plt.show()




########################################################################################
#               KNN
########################################################################################


f1_array_train = []
f1_array_test = []
array_k_values = list(range(1, 100))
for k in array_k_values:
    clf = KNN(n_neighbors = k)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))



plt.plot(array_k_values, f1_array_train, color = 'g', label = "Training data")
plt.plot(array_k_values, f1_array_test, color = 'r', label = "Test data")
plt.title("Mushroom Classifcation: k-nearest neighbors")
plt.xlabel("Number of Neighbors")
plt.ylabel("F1 Score")
plt.legend()
plt.show()


clf = KNN(n_neighbors = 5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print(f1_score(y_test,pred))

train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.01, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    np.random.seed(207888)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("Mushroom Classifcation: CV KNN, K= 5")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()


plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["edible","poisonous"])
plt.title("Mushroom Classifcation: Confusion Matrix")
plt.show()









########################################################################################
#               NEURAL NETWORKS
########################################################################################


f1_array_train = []
f1_array_test = []
hidden_layer_size = list(range(1, 100,10))
for size in hidden_layer_size:
    clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (size,), random_state = 2047)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))



plt.plot(hidden_layer_size, f1_array_train, color = 'g', label = "Training data")
plt.plot(hidden_layer_size, f1_array_test, color = 'r', label = "Test data")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("F1 Score")
plt.title("Mushroom Classification: Neural Networks")
plt.legend()



plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["edible","poisonous"])
plt.show()





clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (20,), random_state = 2047)
clf = clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))


train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.01, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    np.random.seed(207888)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("Mushroom Classifcation: CV Neural Network")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["edible","poisonous"])
plt.title("Mushroom Classifcation: Confusion Matrix")
plt.show()














LE = LabelEncoder()

survey_response = pd.read_csv("C:/Users/Joey/Downloads/archive(1)/responses.csv")

survey_response.describe()
survey_response = survey_response.drop(columns={"House - block of flats"})


null_count = survey_response.isnull().sum()
print(max(null_count))

target = 'Village - town'
survey_response[target] = ['rural' if x == "village" else 'city' for x in survey_response[target] ]
survey_response = survey_response[survey_response[target].notnull()]

survey_response[target].value_counts().plot(kind='bar',
                                    figsize=(12,6),
                                    title= "Count of Rural and City")


survey_response['Punctuality'] = LE.fit_transform(survey_response['Punctuality'].tolist())
survey_response['Lying'] = LE.fit_transform(survey_response['Lying'].tolist())
survey_response['Internet usage'] = LE.fit_transform(survey_response['Internet usage'].tolist())
survey_response['Left - right handed'] = LE.fit_transform(survey_response['Left - right handed'].tolist())
survey_response['Smoking'] = LE.fit_transform(survey_response['Smoking'].tolist())
survey_response['Alcohol'] = LE.fit_transform(survey_response['Alcohol'].tolist())
survey_response['Gender'] = LE.fit_transform(survey_response['Gender'].tolist())
survey_response['Education'] = LE.fit_transform(survey_response['Education'].tolist())
survey_response['Only child'] = LE.fit_transform(survey_response['Only child'].tolist())
survey_response['Village - town'] = LE.fit_transform(survey_response['Village - town'].tolist())


survey_response = survey_response.dropna()
ros = imblearn.over_sampling.RandomOverSampler(random_state = 2047)
ros.fit(survey_response.iloc[:,:-1], survey_response[target])
X, Y = ros.fit_resample(survey_response.iloc[:,:-1], survey_response[target])


temp = pd.DataFrame(pd.DataFrame(Y)[target].value_counts())
temp = temp.reset_index()
temp  = temp.rename(index={0:'city', 1:'rural'})
temp = temp.drop(columns = {'index'})
temp.plot(kind='bar',figsize=(12,6),
                                    title= "Count of Rural and City")


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state  = 2047)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

###########################################################################################################################
#
#
#
#       PART 2: Rural City Classifcation
#
#
#
#
##########################################################################################################################
clf = tree.DecisionTreeClassifier(random_state=2047)
clf = clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(accuracy_score(y_test,pred))

# Note that the main splits don't include the highly correlated variables, may indicate some colinearlity issues in the data
#plt.figure(figsize=(30,10))
#tree.plot_tree(clf, fontsize = 10, feature_names = mushroom_coded.iloc[:,2:].columns, filled = True, class_names = ["edible","poisonous"])
#plt.show()

# 
max_depth = []
entropy_accuracy_train = []
entropy_accuracy_test = []

for i in range(1,10):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = i, random_state = 2047)
    clf.fit(x_train,y_train)
    entropy_accuracy_train.append(f1_score(y_train, clf.predict(x_train)))
    entropy_accuracy_test.append(f1_score(y_test, clf.predict(x_test)))
    max_depth.append(i)
    
# Achieving a max a
plt.plot(range(1,10), entropy_accuracy_train, color = 'g', label = "Training data")
plt.plot(range(1,10), entropy_accuracy_test, color = 'r', label = "Test data")
plt.title("City/Rural Classifcation: Max Depth v Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

print("BEST DEPTH: " + str(entropy_accuracy_test.index(max(entropy_accuracy_test)) + 1))
clf = tree.DecisionTreeClassifier(max_depth = entropy_accuracy_test.index(max(entropy_accuracy_test)) + 1, random_state = 2047, criterion='entropy')
clf = clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))


train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.1, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    np.random.seed(2047)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("City/Rural Classifcation: CV Decision Tree")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["City","Rural"])
plt.show()



########################################################################################
#               BOOSTED DECISION TREE 
########################################################################################
f1_array_train = []
f1_array_test = []
array_tree_values = list(range(1, 100,10))
for tree_ in array_tree_values:
    clf = GradientBoostingClassifier(n_estimators = tree_, random_state = 2047)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))

plt.plot(array_tree_values, f1_array_train, color = 'g', label = "Training data")
plt.plot(array_tree_values, f1_array_test, color = 'r', label = "Test data")
plt.xlabel("Number of Estimators")
plt.ylabel("F1 Score")
plt.title("Rural/City Classifcation: Gradient Boosting DT ")
plt.legend()
plt.show()



tree_boost = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = {'max_depth': [1,2,3,4,5,6],'n_estimators': array_tree_values,'learning_rate':np.linspace(.001,1,5)}, cv = 5)

tree_boost.fit(x_train,y_train)

print(tree_boost.best_params_)


# FIND BEST RESULTS
gbc_best = GradientBoostingClassifier(max_depth =tree_boost.best_params_['max_depth'], n_estimators =tree_boost.best_params_['n_estimators'], learning_rate =tree_boost.best_params_['learning_rate'], random_state = 2047)



clf = gbc_best.fit(x_train, y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))


train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.1, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    np.random.seed(207888)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("City/Rural Classifcation: Cross Validation Gradient Boosting DT ")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["City","Rural"])
plt.show()
##############################################################################
#                            SVM
#################################################################################

f1_array_train = []
f1_array_test = []
kernel_type = ["linear", "rbf", "poly"]
for i in kernel_type:
    clf = SVC(kernel = i, random_state = 2047)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))


plt.plot(kernel_type, f1_array_train, color = 'g', label = "Training data")
plt.plot(kernel_type, f1_array_test, color = 'r', label = "Test data")
plt.xlabel("Kernel Type")
plt.ylabel("F1 Score")
plt.title("City/Rural Classifcation: Support Vector Machines")
plt.legend()
plt.show()



clf = SVC(random_state = 2047, kernel = 'linear')
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print(f1_score(y_test,pred))


train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.1, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    random.seed(2047)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("City/Rural Classifcation: CV SVM, Linear")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

clf.fit(x_test,y_test)
plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["City","Rural"])
plt.title("City/Rural Classifcation: Confusion Matrix")
plt.show()







########################################################################################
#               KNN
########################################################################################


f1_array_train = []
f1_array_test = []
array_k_values = list(range(1, 100))
for k in array_k_values:
    clf = KNN(n_neighbors = k)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))



plt.plot(array_k_values, f1_array_train, color = 'g', label = "Training data")
plt.plot(array_k_values, f1_array_test, color = 'r', label = "Test data")
plt.title("City/Rural: k-nearest neighbors")
plt.xlabel("Number of Neighbors")
plt.ylabel("F1 Score")
plt.legend()
plt.show()


clf = KNN(n_neighbors = 5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print(f1_score(y_test,pred))



train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.1, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    random.seed(2047)
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("City/Rural Classifcation: CV KNN, K= 5")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()


plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["City","Rural"])
plt.title("City/Rural Classifcation: Confusion Matrix")
plt.show()


########################################################################################
#               NEURAL NETWORKS
########################################################################################


f1_array_train = []
f1_array_test = []
hidden_layer_size = list(range(1, 100,10))
for size in hidden_layer_size:
    clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (size,), random_state = 2047)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train)))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test)))



plt.plot(hidden_layer_size, f1_array_train, color = 'g', label = "Training data")
plt.plot(hidden_layer_size, f1_array_test, color = 'r', label = "Test data")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("F1 Score")
plt.title("City/Rural Classification: Neural Networks")
plt.legend()



plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["City","Rural"])
plt.show()





clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (75,), random_state = 2047)
clf = clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(f1_score(y_test,pred))


train_cv_scores = []
test_cv_scores= []

training_sizes =(np.linspace(.1, 1.0, 10)* len(y_train)).astype('int')  
for i in training_sizes:
    index = np.random.randint(x_train.shape[0], size= i)
    new_train_x = x_train.iloc[index, :]
    new_train_y = y_train.iloc[index]
    cv_scores = cross_validate(clf, new_train_x, new_train_y, cv = 10, n_jobs = -1, return_train_score = True,scoring='f1' )
    train_cv_scores.append(np.mean(cv_scores['train_score']))
    test_cv_scores.append(np.mean(cv_scores['test_score']))

# plotting cv results
plt.figure()
plt.plot(training_sizes, train_cv_scores, color = 'g', label = "Training Score")
plt.plot(training_sizes, test_cv_scores, color = 'r', label = "CV Score")
plt.title("City/Rural Classifcation: CV Neural Network")
plt.xlabel("Training Sizes")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

plt.figure()
plot_confusion_matrix(clf, x_test, y_test, display_labels = ["City","Rural"])
plt.title("City/Rural Classifcation: Confusion Matrix")
plt.show()




