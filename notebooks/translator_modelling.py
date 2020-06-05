import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textacy
import re
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import label_binarize



class Modelling:

    def __init__(self, df, model=None):
        self.df = df
        self.model = model
        self.model_gs = None

    def feature_select(self, basic_counts=True, vader=False, pos_counts=False,
                   words=False, adv=False, adj=False):
        '''create column list depending on features to include in the modelling'''
        columns = []
        if basic_counts:
            columns += [i for i in self.df.columns if i.startswith('n_') & i.endswith('_norm')]
        if vader:
            columns += [i for i in self.df.columns if i.startswith('vader_')]
        if pos_counts:
            columns += [i for i in self.df.columns if i.endswith('_count_norm')]
        if words:
            columns += [i for i in self.df.columns if i.endswith('_w')]
        if adj:
            columns += [i for i in self.df.columns if i.endswith('_adj')]
        if adv:
            columns += [i for i in self.df.columns if i.endswith('_adv')]
        return columns

    def drop_features(self, original_list, columns_to_drop):
        return [x for x in original_list if x not in columns_to_drop]

    def modelling_prep(self, predictor_cols, target_col):
#     set predictor and target variables
        X = self.df[predictor_cols]
        y = self.df[target_col]
#     perform train test split, including original indices before shuffling
        indices = list(self.df.index)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=1)
#     normalise the predictor variables
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        return X_train, X_test, y_train, y_test, idx_train, idx_test

    def gridsearch(self, params, X_train, y_train):
    # instantiate model
        self.model_gs = GridSearchCV(estimator=self.model,
                        param_grid=params,
                        cv=5,
                        scoring='accuracy',
                        verbose=1,
                        n_jobs=2,
                        return_train_score=True)
    # fit the model
        self.model_gs.fit(X_train, y_train)
        return self.model_gs

    def gridsearch_score(self, X_train, y_train, X_test, y_test):
        # print the grid search results and store as a dictionary
        results_dict = {}
        results_dict['Best_Parameters'] = self.model_gs.best_params_
        results_dict['Best_CV_Score'] = self.model_gs.best_score_
        results_dict['Best_Train_Score'] = self.model_gs.score(X_train, y_train)
        results_dict['Best_Test_Score'] = self.model_gs.score(X_test, y_test)

        print('Best Parameters:')
        print(results_dict['Best_Parameters'])
        print('Best estimator mean cross validated training score:')
        print(results_dict['Best_CV_Score'])
        print('Best estimator score on the full training set:')
        print(results_dict['Best_Train_Score'])
        print('Best estimator score on the test set:')
        print(results_dict['Best_Test_Score'])
        print('ROC-AUC score on the test set:')

        y_bin = label_binarize(y_test, self.model_gs.classes_)
        for i, class_ in enumerate(self.model_gs.classes_):
            print('Class {}:'.format(class_), round(roc_auc_score(y_bin[:,i],self.model_gs.predict_proba(X_test)[:,i]),2))
        results_dict['AUC_Class_0'] = roc_auc_score(y_bin[:,0],self.model_gs.predict_proba(X_test)[:,0])
        results_dict['AUC_Class_1'] = roc_auc_score(y_bin[:,1],self.model_gs.predict_proba(X_test)[:,1])
        results_dict['AUC_Class_2'] = roc_auc_score(y_bin[:,2],self.model_gs.predict_proba(X_test)[:,2])
        predictions = self.model_gs.predict(X_test)
        results_dict['conmat'] = confusion_matrix(
            y_test, predictions, labels=[0, 1, 2])

        return results_dict

    def save_model(self, out_full_path):
        # save pickle
        with open(out_full_path, 'wb') as fp:
            pickle.dump(self.model_gs, fp)
