import pickle
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from time import time

from helper_functions import dict_ttsplit
from helper_functions import targetFeatureSplit

### read in data, split into train and hold out test set, save to file ######################################################
with open("enron_dataset_modify.pkl", "r") as f:
    data_dict = pickle.load(f)
    
train_dict, test_dict = dict_ttsplit(data_dict)
with open("enron_dataset_train.pkl", "w") as f:
    pickle.dump(train_dict, f)
with open("enron_dataset_test.pkl", "w") as f:
    pickle.dump(test_dict, f)
print """PROCESS: data dictionary split to train and test dictionaries and were saved to 'enron_dataset_train.pkl'
and 'enron_dataset_test.pkl'.\n"""    
    
### remove features which contains more than 50% of null values #############################################################
missing_values = dict()
for person, feature_dict in train_dict.items():
    for feature, values in feature_dict.items():
        if feature not in missing_values:
            missing_values[feature] = 0
        if values == "NaN":
            missing_values[feature] += 1

label_name = ["poi"]
features_names = []
exclude = ["poi", "email_address"]
for feature in missing_values:
    if missing_values[feature]<58 and feature not in exclude:
        features_names.append(feature)

        
### get features and label ##################################################################################################
labels, features_v1 = targetFeatureSplit(train_dict, label_name+features_names)


### for the training set, do a ANOVA test and get the feature ranking ########################################################
F_score_v1, P_v1 = f_classif(features_v1, labels)
ranking_v1 = zip(F_score_v1, P_v1, features_names)
ranking_v1.sort()
print "First Time ANOVA Test:\n"
print "%34s %10s %10s" % ("feature name", "F_score", "P_value")
for f, p, feature in ranking_v1:
    print "%34s %10.3f %10.3f" % (feature, f, p)
print "\n"

### create new keys 'fraction_from_poi_to_this_person', 'fraction_from_this_person_to_poi' ################
for person, feature_dict in train_dict.items():
    if feature_dict["from_poi_to_this_person"] != "NaN" and feature_dict["from_messages"] != "NaN":
        feature_dict["fraction_from_poi_to_this_person"] \
        = feature_dict["from_poi_to_this_person"] * 1.0/feature_dict["from_messages"]
    else:
        feature_dict["fraction_from_poi_to_this_person"]="NaN"
        
    if feature_dict["from_this_person_to_poi"] != "NaN" and feature_dict["to_messages"] != "NaN":
        feature_dict["fraction_from_this_person_to_poi"] \
        = feature_dict["from_this_person_to_poi"] * 1.0/feature_dict["to_messages"]
    else:
        feature_dict["fraction_from_this_person_to_poi"]="NaN"

with open("enron_dataset_train_modify.pkl", "w") as f:
    pickle.dump(train_dict, f)
    
print "PROCESS: new features added, saved to 'enron_dataset_train_modify.pkl'.\n"


### add the new feature to feature list ##################################################################
features_names.append("fraction_from_poi_to_this_person")
features_names.append("fraction_from_this_person_to_poi")


### repeat the above step get training set and hold out testing set, and use ANOVA to rank the features
labels, features_v2 = targetFeatureSplit(train_dict, label_name+features_names)
F_score_v2, P_v2 = f_classif(features_v2, labels)
ranking_v2 = zip(F_score_v2, P_v2, features_names)
ranking_v2.sort()
print "Second Time ANOVA Test:\n"
print "%34s %10s %10s" % ("feature name", "F_score", "P_value")
for f, p, feature in ranking_v2:
    print "%34s %10.3f %10.3f" % (feature, f, p)
print"\n"

### use RFECV nested with GridSearchCV to select features ################################################### 

print "Using RFECV nested with GridSearchCV to do Feature Selection:\n"

t_rfecv = time()
param_grid ={'estimator__n_estimators': [10, 20, 30],
             'estimator__min_samples_split': [1, 2, 5],
             'estimator__max_features':[0.5, 0.7, 1.0]}
estimator = RandomForestClassifier(random_state=42)
selector = RFECV(estimator, step=1, cv=5, scoring='f1')
clf = GridSearchCV(selector, param_grid, cv=5, scoring='f1')
clf.fit(features_v2, labels)


### print feature ranking
print "Running Time: %0.3fs\n" % (time()-t_rfecv)
print "Feature Ranking:\n"
RFECV_ranking = zip(clf.best_estimator_.ranking_, features_names)
RFECV_ranking.sort()
for ranking, feature in RFECV_ranking:
    print "%34s %10s" % (feature, ranking)
print "\n"
### print best estimator
print "Best Estimator:\n\n", clf.best_estimator_.estimator_
print "\n"
# Plot number of features VS. cross-validation scores
plt.figure()
plt.title("Number of Features vs Cross Validation Scores")
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross Validation Score")
plt.plot((range(1, len(clf.best_estimator_.grid_scores_)+1)), clf.best_estimator_.grid_scores_)
plt.show()

### based on results above, select three combinations of features for further testing ##############

# select features which get p<1% in ANOVA test
sf1 = ["poi", "total_stock_value", "exercised_stock_options", "bonus", "restricted_stock", "salary", 
                     "shared_receipt_with_poi", "total_payments", "expenses"]

# select features from RFECV experiment
sf2 = ["poi", "bonus", "from_poi_to_this_person", "expenses", "exercised_stock_options"]

# a combination of features which have the highest rank in ANOVA test, together with features from RFECV experiment
sf3 = ["poi", "bonus", "from_poi_to_this_person", "expenses", "total_stock_value", "exercised_stock_options"]

sfs_dict = {"sf1":sf1, "sf2":sf2, "sf3":sf3}

with open("initial_three_combinations_of_features.pkl", "w") as f:
    pickle.dump(sfs_dict, f)

print "PROCESS: initial selection of three combination of features saved to 'initial_three_combinations_of_features.pkl'"
