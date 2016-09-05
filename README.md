# Enron\_POI\_Detector

## 1. Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. This project aims to build a person of interest (POI) identifier which detect persons who were indicted, reached a settlement or plea deal with government, or testified in exchange for prosecution immunity.

## 2. References
* Enron email dataset: "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
* Enron insiderpay: enron61702insiderpay.pdf
* Udacity's Intro to Machine Learning Course 

## 3. Preprocessing Details (done by Udacity, used as course material)
* final_project_dataset.pkl
As a preprocessing to this project, the Enron email and financial data were combined into a dictionary (dumped into final_project_dataset.pkl), where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features are listed as following:  
['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'email_address', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'directror_fees']  
These fall into three major types of features, namely financial features, email features and POI labels.  
financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)
email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)  
POI label: [‘poi’] (boolean, represented as integer)
