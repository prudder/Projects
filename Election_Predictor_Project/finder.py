from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import ElasticNetCV, ElasticNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modality import modeler as md
import warnings
warnings.filterwarnings("ignore")

ced_df = pd.read_csv('./Database/CED_model.csv')

#defining the feature that will be excluded for the predictor matrix
feats_to_excl = ['divisionnm','divisionid','stateab','partyab','lncvotes','lncpercentage','alpvotes','alppercentage',
                 'swing','election_year','candidateid','givennm','surname','partynm','enrolment','turnout',
                 'turnoutpercentage','turnoutswing','totalpercentage','closeofrollsenrolment','notebookrolladditions',
                 'notebookrolldeletions','reinstatementspostal','reinstatementsprepoll','reinstatementsabsent',
                 'reinstatementsprovisional','year','ced','ced_state','census_year']

data = ced_df.drop(columns=feats_to_excl)

X = data[[col for col in data.columns if 'is_right' not in col]]
y = data['is_right']

baseline = y.value_counts(normalize=True)[1]


lr = LogisticRegression(C=0.1,penalty='l2',solver='newton-cg')

#Defining the training X and Y's, all rows which are not election year 2019
X_train = ced_df[ced_df['year'] != 2019][[col for col in ced_df.columns if 'is_right' not in col]]
X_train = X_train.drop(columns=feats_to_excl)
y_train = ced_df[ced_df['year'] != 2019]['is_right']

#Defining the testing X and Y's, all rows which are election year 2019
X_test = ced_df[ced_df['year'] == 2019][[col for col in ced_df.columns if 'is_right' not in col]]
X_test = X_test.drop(columns=feats_to_excl)
y_test = ced_df[ced_df['year'] == 2019]['is_right']

lr_mod = lr.fit(X_train,y_train)

y_pred = lr_mod.predict(X_test)

probabs = md.probability_table(lr_mod,X_test,y_test,ref_df=ced_df)

#Putting the coefficients into a data frame, zipped up with the feature names
c_coefs_lr = pd.DataFrame(dict(zip(X_train.columns,lr_mod.coef_[0])),index=['Value']).T
c_coefs_lr['ABS_Value'] = c_coefs_lr['Value'].apply(abs)
#Applying the logit convert function to get the probabilities
c_coefs_lr['Probability'] = c_coefs_lr['Value'].apply(md.logit_convert)
c_coefs_lr.sort_values(by='Value',ascending=False)

md.finder(probabs,ced_df,c_coefs_lr)