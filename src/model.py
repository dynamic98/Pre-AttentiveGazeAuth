import os
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from tqdm import tqdm
import json
from .load_data import load_data
from .preprocessing import data_preprocessing

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, data):
        self.zeroR = DummyClassifier(strategy='most_frequent', random_state=0)  # ZeroR
        self.DT = DecisionTreeClassifier(random_state=0) # Decision Tree
        self.KNN = KNeighborsClassifier() # K-Nearest Neighbors
        self.NB = GaussianNB() # Naive Bayes
        self.SVM = SVC(random_state=0) # Support Vector Machine
        self.LR = LogisticRegression(random_state=0) # Logistic Regression
        self.AB = AdaBoostClassifier(random_state=0) # AdaBoost
        self.RF = RandomForestClassifier(random_state=0) # Random Forest
        
        self.clf_names = ['ZR', 'DT', 'KNN', 'NB', 'SVM', 'LR', 'AB', 'RF']
        self.clf_names_dict = {'ZR': self.zeroR, 'DT': self.DT, 'KNN': self.KNN, 'NB': self.NB, 
                               'SVM': self.SVM, 'LR': self.LR, 'AB': self.AB, 'RF': self.RF}
        self.x_domain_exclude = ['participant', 'session', 'stimuli_index']
        self.data = data
        
        
    def cross_validation(self, cv=5, *clf_list):
        # cv is the number of folds
        # clf_list is a list of classifier names to be used
        # clf_list should be consisted of the names in self.clf_names (e.g. 'ZR', 'DT', 'KNN', 'NB', 'SVM', 'LR', 'AB', 'RF')
        kf = StratifiedKFold(n_splits=cv)
        x_data, y_data = self.take_x_y()
        x_data = x_data.to_numpy()
        y_data = y_data.to_numpy()
        
        results = self.make_empty_results(cv, clf_list)
        
        for i, (train, test) in enumerate(kf.split(x_data, y_data)):
            print(f'Fold {i+1} is in progress...')
            for clf_name in tqdm(clf_list):
                clf = clone(self.clf_names_dict[clf_name])
                clf.fit(x_data[train], y_data[train])
                predictions = clf.predict(x_data[test])
                results[clf_name][f'{i+1}']['accuracy'].append(accuracy_score(y_data[test], predictions))
                results[clf_name][f'{i+1}']['precision'].append(precision_score(y_data[test], predictions, average='weighted'))
                results[clf_name][f'{i+1}']['recall'].append(recall_score(y_data[test], predictions, average='weighted'))
                results[clf_name][f'{i+1}']['f1'].append(f1_score(y_data[test], predictions, average='weighted'))

        return results
    
    def leave_one_session_out_cross_validation(self, *clf_list):
        results = self.make_empty_results(5, clf_list)
        for session in range(1,6):
            train_x, train_y = self.take_x_y(self.data[self.data['session'] != session])
            test_x, test_y = self.take_x_y(self.data[self.data['session'] == session])
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            test_x = test_x.to_numpy()
            test_y = test_y.to_numpy()
            print(f'Session {session} is in progress...')
            for clf_name in tqdm(clf_list):
                clf = clone(self.clf_names_dict[clf_name])
                clf.fit(train_x, train_y)
                predictions = clf.predict(test_x)
                results[clf_name][f'{session}']['accuracy'].append(accuracy_score(test_y, predictions))
                results[clf_name][f'{session}']['precision'].append(precision_score(test_y, predictions, average='weighted'))
                results[clf_name][f'{session}']['recall'].append(recall_score(test_y, predictions, average='weighted'))
                results[clf_name][f'{session}']['f1'].append(f1_score(test_y, predictions, average='weighted'))
        return results
    
    def make_empty_results(self, cv, clf_list):
        results = {}
        for clf_name in clf_list:
            results[clf_name] = {}
            for i in range(cv):
                results[clf_name][f'{i+1}'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        return results
        
        
    def take_x_y(self, data=pd.DataFrame()):
        if data.empty:
            x_data = self.data.loc[:, ~self.data.columns.isin(self.x_domain_exclude)]
            y_data = self.data['participant']
        else:
            x_data = data.loc[:, ~data.columns.isin(self.x_domain_exclude)]
            y_data = data['participant']
        return x_data, y_data
    
    def set_x_domain_exclude(self, x_domain_exclude:list):
        # x_domain_exclude is a list of column names that should be excluded from the x_data
        self.x_domain_exclude = x_domain_exclude
    
    def get_data(self):
        return self.data

if __name__ == "__main__":
    pass