import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from helper_functions import test_score

"""final test on hold out testing set with optimized features and classifiers"""

### read enron training data, testing data and optimized features and classifiers ##############
with open("enron_dataset_train.pkl", "r") as f:
    train_dict = pickle.load(f)
    
with open("enron_dataset_test.pkl", "r") as f:
    test_dict = pickle.load(f)

with open("final_clfs_features.pkl", "r") as f:
    data = pickle.load(f)

clf_best_precision = data["clf_best_precision"]
clf_best_recall_f1_f2 = data["clf_best_recall_f1_f2"]
labels_features = data["labels_features"]

### train on training set and test on testing set #############################################
accuracy_1, precision_1, recall_1, f1_1 = test_score(train_dict, test_dict, labels_features, clf_best_precision)
accuracy_2, precision_2, recall_2, f1_2 = test_score(train_dict, test_dict, labels_features, clf_best_recall_f1_f2)
print "Result from Classifiers which Give Best Precision or Best Recall/f1/f2 from Cross Validation:\n"

print "classifier which give best precision from cross validation(clf1):"
print clf_best_precision
print "\nclassifier which give best recall/f1/f2 from cross validation(clf2):"
print clf_best_recall_f1_f2

print "\nResults on testing set:"
print "{:^5}{:^12}{:^12}{:^12}{:^12}".format("clf", "accuracy", "precision", "recall", "f1")
print "{:^5}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}".format("clf1", accuracy_1, precision_1, recall_1, f1_1)
print "{:^5}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}".format("clf2", accuracy_2, precision_2, recall_2, f1_2)
