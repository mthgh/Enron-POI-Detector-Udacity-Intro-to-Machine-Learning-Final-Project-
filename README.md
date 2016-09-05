# Enron\_POI\_Detector

## 1. Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. This project aims to build a person of interest (POI) identifier which detect persons who were indicted, reached a settlement or plea deal with government, or testified in exchange for prosecution immunity.

## 2. References
* Enron email dataset: "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
* Enron insiderpay: enron61702insiderpay.pdf
* Udacity's Intro to Machine Learning Course 

## 3. Resources from Udacity Course Material (preprocessing details)
* ```final_project_dataset.pkl```
<br>
As a preprocessing to this project, the Enron email and financial data were combined into a dictionary (dumped into 'final_project_dataset.pkl'), where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features are listed as following:  
['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'email_address', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'directror_fees']  
These fall into three major types of features, namely financial features, email features and POI labels.  
financial features: \['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'\] (all units are in US dollars)     
email features: \['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'\] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)       
POI label: \['poi'\] (boolean, represented as integer)
* ```emails_by_address```
<br>
This directory contains many text files, each of which contains all the messages to or from a particular email address. It is for reference, if more advanced features need to be created, there is no need to process the email corpus.

## 4. Data Exploration
```data_exploration``` was used to investigate the dataset and check for outliers for the financial data.  
Here is an overview of the dataset from ```final_project_dataset.pkl```
<pre>
Total number of persons in dataset: 146
Total number of POI in dataset: 18
Total number of features in dataset: 21
Number of missing values for each feature in dataset:
{'bonus': 64,
 'deferral_payments': 107,
 'deferred_income': 97,
 'director_fees': 129,
 'email_address': 35,
 'exercised_stock_options': 44,
 'expenses': 51,
 'from_messages': 60,
 'from_poi_to_this_person': 60,
 'from_this_person_to_poi': 60,
 'loan_advances': 142,
 'long_term_incentive': 80,
 'other': 53,
 'poi': 0,
 'restricted_stock': 36,
 'restricted_stock_deferred': 128,
 'salary': 51,
 'shared_receipt_with_poi': 60,
 'to_messages': 60,
 'total_payments': 21,
 'total_stock_value': 20}
</pre>
To check for outliers in financial data, all financial features were extracted from the dataset. Using dimension reduction (PCA), the multidimensional financial features were projected to two dimensions and the mapping was visualized (Figure 1).   
This visualization clearly indicate two outliers. A further exploration of the outliers revealed one is from "TOTAL" value and the other is from financial data of "LAY KENNETH L". The formal outlier was removed as it was not a valid data point, whereas the latter was kept because it is a valid input, and it is actually coming from a "POI"  
After removing the invalid data point, the outlier checking process was repeated (extract all financial features and apply PCA to project to two dimensions) and several more outliers were observed (Figure 2), the new outliers were all valid financial data from 'HIRKO JOSEPH', 'RICE KENNETH D', 'SKILLING JEFFREY K' and 'PAI LOU L'. Therefore they are kept without change.  
The modified dataset were dumped into ```final_project_dataset_modify.pkl```
