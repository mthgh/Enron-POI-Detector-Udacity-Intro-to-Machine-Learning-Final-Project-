import pickle
from pprint import pprint
import numpy as np

from helper_functions import targetFeatureSplit
from helper_functions import dimention_reduction
from helper_functions import draw_2d_financial


### Load the dictionary containing the dataset #################################################################
with open("enron_dataset.pkl", "r") as f:
    data_dict = pickle.load(f)
persons = data_dict.keys()

### data overview ##############################################################################################
print "OVERVIEW of DATASET\n"
print "Total number of persons in dataset:", len(data_dict)
print "Total number of POI in dataset:", sum( feature_dict["poi"] for person, feature_dict in data_dict.items())

missing_values = dict()
for person, feature_dict in data_dict.items():
    for feature, values in feature_dict.items():
        if feature not in missing_values:
            missing_values[feature] = 0
        if values == "NaN":
            missing_values[feature] += 1

print "Total number of features in dataset:", len(missing_values)
print "Number of missing values for each feature in dataset:" 
pprint(dict(missing_values))
print "\n"

### investigate outliers #######################################################################################
print "OUTLIER IVESTIGATION\n"

### financial keys in data_dict ################################################################################
financial_list = ["salary", "deferral_payments", "total_payments", "loan_advances", "bonus",
                  "restricted_stock_deferred", "deferred_income", "total_stock_value", "expenses",
                  "exercised_stock_options","other", "long_term_incentive", "restricted_stock", "director_fees"]

### extract financial_data as nxk arrays (n: no. of datapoints; k: no. of financial features) ##################
financial_data = targetFeatureSplit(data_dict, financial_list, remove_all_zeroes=False)

### dimentionality reduction and visualization ##################################################################
trans_financial_data = dimention_reduction(financial_data)
draw_2d_financial(trans_financial_data, "First Exploration for Outliers")

### two obvious outliers pop up, check the ourliers #############################################################
max_index = trans_financial_data.argmax(axis=0)
print "1st outlier with person name:", persons[max_index[0]]
print "2st outlier with person name:", persons[max_index[1]]
print """The 1st outlier is for the total financial data, should be removed, while
the 2nd outlier is a resonable data, it is from POI, should be kept in dataset."""

### remove first outlier #########################################################################################
data_dict.pop("TOTAL")
persons_v1 = data_dict.keys()
print "\nPROCESS: The 1st outlier with person name 'TOTAL' removed.\n"

### repeat outlier checking process to see if there are other outliers ###########################################
financial_data_v1 = targetFeatureSplit(data_dict, financial_list, remove_all_zeroes=False)
trans_financial_data_v1 = dimention_reduction(financial_data_v1)
draw_2d_financial(trans_financial_data_v1, "Second Exploration for Outliers")
inverse_sorted_index = trans_financial_data_v1.argsort(axis=0)[::-1]
print "Some other outliers were observed after removing 'TOTAL', the person names are as following:"
print persons_v1[inverse_sorted_index[0][1]]
print persons_v1[inverse_sorted_index[1][1]]
print persons_v1[inverse_sorted_index[2][1]]
print persons_v1[inverse_sorted_index[3][1]]
print persons_v1[inverse_sorted_index[0][0]]
print "These outliers were kept in the dataset since they are valid data.\n"

### save revised data_dict to file #################################################################################
with open("enron_dataset_modify.pkl", "w") as f:
    pickle.dump(data_dict, f)
print "PROCESS: modified dataset saved to 'enron_dataset_modify.pkl'."
