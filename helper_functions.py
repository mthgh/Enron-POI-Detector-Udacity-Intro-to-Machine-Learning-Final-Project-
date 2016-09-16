import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


### dimentionality reduction function (used in data exploration and outlier removal)######################################

def dimention_reduction(financial_data):
    pca = RandomizedPCA(n_components=2)
    pca = pca.fit(financial_data)
    trans_financial_data = pca.transform(financial_data)
    return trans_financial_data


### visualization function (used in data exploration and outlier removal)#################################################

def draw_2d_financial(financial_data, title):
    plt.figure()
    plt.scatter(financial_data[:, 0], financial_data[:, 1])
    plt.xlabel("pca_primary")
    plt.ylabel("pca_secondary")
    plt.title(title)
    plt.show()



######## split the data dictionary into hold out testing data and training data ###########################################


def dict_ttsplit(data_dict):
    labels = [int(data_dict[person]["poi"]) for person in data_dict]
    cv = StratifiedShuffleSplit(labels, n_iter=1, test_size=0.2, random_state=42)
    train_dict = dict()
    test_dict = dict()
    persons = data_dict.keys()
    for train_idx, test_idx in cv:
        for ii in train_idx:
            person = persons[ii]
            train_dict[person] = data_dict[person]
        for jj in test_idx:
            person = persons[jj]
            test_dict[person] = data_dict[person]
    return train_dict, test_dict



###############targetFeatureSplit##########################################################################################
    """ 
    A general tool for converting data from the
    dictionary format to an (n x k) python list 

    n--no. of key-value pairs in dictonary
    k--no. of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, where each key-
    value pair in the dict is the name
    of a feature, and its value for that person

    if the first item in feature_list is "poi", the function
    returns two python lists: labels ("poi" as the target) and features.
    for example, if you want to have the poi label as the target,
    and the features you want to use are the person's
    salary and bonus, here's what you would do:

    feature_list = ["poi", "salary", "bonus"] 
    label, features = targetFeatureSplit(data_dictionary, feature_list)

    if the first item in feature_list is not "poi", the function
    returns one numpy array (n x k) of the features.
    """

def targetFeatureSplit( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False):
    """ 
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which any of the features you seek are 0.0
        NOTE: first feature is assumed to be 'poi' and is not checked for removal for zero or missing values.
    """

    return_list = []


    keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    if features[0] != "poi":
        return np.array(return_list)
    
    labels_data = []
    features_data = []
    for item in np.array(return_list):
        labels_data.append(item[0])
        features_data.append(item[1:])
        

    return labels_data, features_data



######## cross validation test function ###################################################################

def estimator_cv(data_dictionary, labels_and_features_name, clf, n_iter=100, test_size=0.1):
    labels, features = targetFeatureSplit(data_dictionary, labels_and_features_name)
    cv_iter = StratifiedShuffleSplit(labels, n_iter, test_size, random_state=42)
    
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    
    for train_idx, test_idx in cv_iter:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
            
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        for prediction, truth in zip(pred, labels_test):
            if prediction==0 and truth==0:
                true_negatives += 1
            elif prediction==0 and truth==1:
                false_negatives += 1
            elif prediction==1 and truth==0:
                false_positives += 1
            elif prediction==1 and truth==1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        
        return accuracy, precision, recall, f1, f2
    
    except:
        
        return accuracy, 0, 0, 0, 0
        
#         print "Estimator:", clf, "\n"
#         print "Total Predictions:%s, True Positives:%s, False Positives:%s, True Negatives:%s, False Negatives:%s" \
#         % (total_predictions, true_positives, false_positives, true_negatives, false_negatives)
#         print "%10s %0.3f" % ("Accuracy:", accuracy)
#         print "%10s %0.3f" % ("Precision:", precision)
#         print "%10s %0.3f" % ("Recall:", recall)
#         print "%10s %0.3f" % ("f1:", f1)
#         print "%10s %0.3f" % ("f2:", f2)
#     except:
#         print "Got a divide by zero when trying out:", clf
#         print "Precision or recall may be undefined due to a lack of true positive predicitons."



########## find best clf based on giving scoreing method #################################################

def find_best(scoring, score_grid):
    
    score_index_map = {"accuracy":0,
                      "precision":1,
                      "recall":2,
                      "f1":3,
                      "f2":4}
    if scoring not in score_index_map:
        print "Not valid scoring method"
        return
    score_index = score_index_map[scoring]
    
    scoring_list = []
    for clf, scores in score_grid.items():
        scoring_list.append(scores[score_index])
    best_index = np.array(scoring_list).argmax()
    best_clf = score_grid.keys()[best_index]
    return best_clf

### testing function on testing set ###########################################
def test_score(train_dict, test_dict, labels_features, clf):
    label_test, feature_test =  targetFeatureSplit(test_dict, labels_features)
    label_train, feature_train = targetFeatureSplit(train_dict, labels_features)
    clf = clf.fit(feature_train, label_train)
    predictions = clf.predict(feature_test)
    accuracy = accuracy_score(label_test, predictions)
    precision, recall, f1, unused = precision_recall_fscore_support(label_test, predictions, pos_label=1, average="binary")
    return accuracy, precision, recall, f1
