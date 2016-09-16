import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from helper_functions import estimator_cv
from helper_functions import find_best

### set up different classifiers for cross validation #####################################################

estimators_svc = [('feature_scale', MinMaxScaler()), ('svm', SVC())]
clf_svc = Pipeline(estimators_svc)
estimators_knn = [('feature_scale', MinMaxScaler()), ('knn', KNeighborsClassifier())]
clf_knn = Pipeline(estimators_knn)

clf_dict = {"GaussianNB()":GaussianNB(), 
            "DecisionTreeClassifier()":DecisionTreeClassifier(), 
            "RandomForestClassifier()":RandomForestClassifier(), 
            "AdaBoostClassifier()":AdaBoostClassifier(),
            "SVC()":clf_svc, 
            "KNeighborsClassifier()":clf_knn}

### read in training data and three different combinations of features for further selection #############
with open("initial_three_combinations_of_features.pkl", "r") as f:
    sfs_dict = pickle.load(f)
with open("enron_dataset_train.pkl") as f:
    train_dict = pickle.load(f)

### cross validation with different classifiers ##########################################################
screening_dict = {}
for clf_string, clf in clf_dict.items():
    for sf_string, sf in sfs_dict.items():
        accuracy, precision, recall, f1, f2 = estimator_cv(train_dict, sf, clf, n_iter=100, test_size=0.1)
        if clf_string not in screening_dict:
            screening_dict[clf_string]={}
        screening_dict[clf_string][sf_string]=[accuracy, precision, recall, f1, f2]

### print scores from cross validation using different classifiers ########################################

print "Screening Different Classifiers:\n"

# title
print '{:^30}{:^12}{:^12}{:^12}{:^12}{:^12}{:^12}'\
.format('classifier', "feature", "accuracy", "precision", "recall", "f1", "f2")
print "-"*100

# result in each classifier
for classifier in screening_dict:
    for sf, numbs in screening_dict[classifier].items():
        print '{:^30}{:^12}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}'\
        .format(classifier, sf, numbs[0], numbs[1], numbs[2], numbs[3], numbs[4])
    print "-"*100
    
### use AdaBoost as the classifier, sf3 as features, tune the parameters using cross validation #############
param_grid = {
              "min_samples_split" : [2, 5, 8],
              "max_features":[0.5, 0.8, 1.0],
              "n_estimators": [30, 50, 80, 100]
             }
sf_in_use = sfs_dict["sf3"]

score_grid = {}
for msp in param_grid["min_samples_split"]:
    for mf in param_grid["max_features"]:
        for ne in param_grid["n_estimators"]:
            base_estimator = DecisionTreeClassifier(min_samples_split=msp, max_features=mf, random_state=42)
            clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=ne, random_state=42)
            accuracy, precision, recall, f1, f2 = estimator_cv(train_dict, sf_in_use, clf, n_iter=100, test_size=0.1)
            score_grid[clf]=[accuracy, precision, recall, f1, f2]
for ne in param_grid["n_estimators"]:
    clf = AdaBoostClassifier(n_estimators=ne)
    accuracy, precision, recall, f1, f2 = estimator_cv(train_dict, sf_in_use, clf, n_iter=100, test_size=0.1)
    score_grid[clf]=[accuracy, precision, recall, f1, f2]

### print out AdaBoost parameters tuning result ###############################################################
print "\nAdaBoost Tuning:"
print """Note: if 'min_samples_split' and 'max_featues' are None, base_estimator = None.
Else, base_estimator = DecisionTreeClassifier(min_samples_split = min_samples_split, 
max_features = max_features)"""
print "\n"

print "{:^17}{:^16}{:^16}{:^16}{:^11}{:^11}{:^11}{:^11}{:^11}"\
.format("", "n_estimator","min_samples_split","max_features","accuracy", "precision", "recall", "f1", "f2")

scoring_methods = ["accuracy", "precision", "recall", "f1", "f2"]
for sm in scoring_methods:
    clf = find_best(sm, score_grid)
    ne = clf.get_params().get("n_estimators")
    msp = clf.get_params().get('base_estimator__min_samples_split')
    mf = clf.get_params().get('base_estimator__max_features')
    accuracy, precision, recall, f1, f2 = score_grid[clf]
    print "{:^17}{:^16}{:^16}{:^16}{:^11.3f}{:^11.3f}{:^11.3f}{:^11.3f}{:^11.3f}"\
    .format("best_"+sm, ne, msp, mf, accuracy, precision, recall, f1, f2)
    
### try PCA, the best estimator above(based on f1 score) and sf3 to do classification #############################
clf_f1 = find_best("f1", score_grid)
sf_in_use = sfs_dict["sf3"]

print "\nPCA Analysis:\n"
print "clf:", clf_f1, "\n"
print "{:^15}{:^12}{:^12}{:^12}{:^12}{:^12}".format("n_components", "accuracy", "precision", "recall", "f1", "f2")

for n in range(1, 6):
    estimators_pca = [('dimension_reduction', PCA(n_components=n)), ('abc', clf_f1)]
    clf_pca = Pipeline(estimators_pca)
    accuracy, precision, recall, f1, f2 = estimator_cv(train_dict, sf_in_use, clf_pca, n_iter=100, test_size=0.1)
    print "{:^15}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}".format(n, accuracy, precision, recall, f1, f2)
    
### final result: sf3 as features, two classifiers ( one give best precision,  ##################################
### the other give best recall/f1/f2 according to cross validation ). ###########################################
### Write them to file ##########################################################################################
labels_features = sfs_dict["sf3"]
clf_best_precision = find_best("precision", score_grid)
clf_best_recall_f1_f2 = find_best("recall", score_grid)

final_data = {"labels_features":labels_features, 
        "clf_best_precision":clf_best_precision, 
        "clf_best_recall_f1_f2":clf_best_recall_f1_f2}

with open("final_clfs_features.pkl", "w") as f:
    pickle.dump(final_data, f) 
    
print """\nPROCESS: final selection of features, clf with best precision from cv, and clf with best recall/f1/f2
from cv saved to 'final_clfs_features.pkl'"""
