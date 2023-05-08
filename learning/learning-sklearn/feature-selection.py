#! /usr/bin/env python

import sys
import os
import io
import numpy as np
import pandas as pd

import sklearn

from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import SVR
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
from random import *
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

import argparse
from collections import defaultdict

from subdominization import helpers





class FeatureSelector():

    def __init__(self, dataset, modelType, njobs=1):
        '''
        Constructor take parameters:
        isBalanced, Boolean for balance the target of prediction in training phase
        modelType = 'LRCV', 'LG', 'RF' , 'SVMCV','NBB', 'NBG'. 'DT';
                       Logistic Regression,
                       Logistic Regression with Cross Validation,
                       Random Forest,
                       Support Vector MAchine with CV grid search,
                       Naive Bayes classifier with Bernoulli estimator
                       Naive Bayes classifier with Gaussian estimator
                       DT is decision Tree
        you ahve to give:
                  'training_file' that is a CSV file containing in each line the feature vectors (validation of rules), and the las column the target to be predicted

        njobs, to paralellice the training phase, default njobs=-1 to get the availables cores
        testSize, is to define the size of test set. Default value calculates the test set random, with a 5% of the training set.
        '''

        self.selector_type = modelType

        # print dataset.shape
        # separate in features an target
        X, y = dataset.iloc[:,:-1], list(dataset.iloc[:, -1])

        # if we want to separate into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00000001, random_state=0)

        X=X_train
        y=y_train

        if y.count(y[0]) == len(y):
            # only one output class
            self.selector_type = "None"
            self.num_features = len(X[0])
            return

        # Standarize features
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

        #self.X_std = scaler.fit_transform(X)
        self.X_std = X

        if (modelType == 'LOGR'):
            # Create decision tree classifer object
            clf = LogisticRegression(random_state=0, class_weight=None)
            self.model = clf.fit(self.X_std, y)
        elif (modelType == "RANSAC"):
            regr = RANSACRegressor()
            self.model = regr.fit(self.X_std, y)
        elif (modelType=='LINR'):
            regr = LinearRegression(n_jobs=njobs)
            self.model = regr.fit(X, y)
        elif (modelType=='LOGRCV'):
            #fold = KFold(len(y), n_folds=20, shuffle=True, random_state=1)
            fold = KFold(n_splits=5, shuffle=True, random_state=1)
            searchCV = LogisticRegressionCV(
                        #Cs = list(np.power(10.0, np.arange(-10, 10))),
                        penalty = 'l2',
                        scoring = 'roc_auc',
                        cv = fold,
                        random_state = 777,
                        max_iter = 100,
                        fit_intercept = True,
                        solver = 'newton-cg',
                        multi_class = "multinomial",
                        tol = 0.00001,#10,
                        class_weight = None,
                        n_jobs = njobs
                    )
            self.model = searchCV.fit(self.X_std , y)
        elif (modelType == 'RF'):
            # set n_jobs to 1 to fix the low CPU usage when computing estimates
            clf_RG = RandomForestClassifier(n_jobs=1, random_state=0, class_weight=None)
            self.model = clf_RG.fit(self.X_std, y)
        elif (modelType == 'SVCCV'):
            classifier_pipeline=None
            #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            #     'C': [1, 10, 100, 1000]},
            #    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            params={'kernel': ['linear'],
                    'C': [1, 5, 10],
                    'degree': [2,3],
                    'gamma': [0.025, 1.0, 1.25, 1.5, 1.75, 2.0],
                    'coef0': [2, 5, 9],
                    'class_weight': [None]}

            clf = GridSearchCV(SVC(probability=True),
                               params,
                               cv=3,
                               scoring='roc_auc',
                               n_jobs=njobs)
            self.model =clf.fit(self.X_std, y)
        elif (modelType=='SVC'):
            clf = SVC(probability=True, class_weight=None, kernel="linear")
            self.model = clf.fit(self.X_std, y)
        elif (modelType=='SVR'):
            clf = SVR(kernel="linear")
            self.model = clf.fit(self.X_std, y)
        elif (modelType=='NBB'):
            # Create decision tree classifer object
            clf= None
            if (not self.isBalanced):
                clf = BernoulliNB()
            else:
                clf= make_pipeline(scaler, BernoulliNB())
            self.model = clf.fit(self.X_std, y)
            self.is_classifier = True
        elif (modelType=='NBG'):
            # Create decision tree classifer object
            clf = GaussianNB()
            self.model = clf.fit(self.X_std, y)
            self.is_classifier = True
        elif (modelType=='DT'):
            clf = tree.DecisionTreeClassifier(class_weight=None)
            self.model = clf.fit(self.X_std, y)
        elif (modelType=='DT_RG'):
            clf = tree.DecisionTreeRegressor()
            self.model = clf.fit(self.X_std, y)
            #get_code(self.model, ["rule" + str(i) for i in range(self.X_std.shape[1])])
        elif (modelType == "KNN"):
            clf = KNeighborsClassifier(n_neighbors=1)
            self.model = clf.fit(self.X_std, y)
            self.is_classifier = True
        elif (modelType == "KNN_R"):
            clf = KNeighborsRegressor(n_neighbors=1)
            self.model = clf.fit(self.X_std, y)
        elif (modelType=='DTGD_RG'):
            clf = GridSearchCV(tree.DecisionTreeRegressor(random_state=0),
                                param_grid={'min_samples_split': range(2, 10)},
                                scoring=make_scorer(r2_score),
                                cv=5,
                                refit=True)
            self.model = clf.fit(self.X_std, y)
        elif (modelType == 'RF_RG'):
            clf = RandomForestRegressor(n_jobs=1)
            self.model  = clf.fit(self.X_std, y)
        elif (modelType == 'RFGD_RG'):
            param_grid = {
            "n_estimators"      : [10,20,30],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }
            clf = GridSearchCV(RandomForestRegressor(), param_grid, n_jobs=1, cv=5)
            self.model = clf.fit(self.X_std, y)
        elif (modelType=='SVRGD'):
            classifier_pipeline=None
            #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            #     'C': [1, 10, 100, 1000]},
            #    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            params={'kernel':['linear'],
                'C':[1, 5, 10],
                'degree':[2,3],
                'gamma':[0.025, 1.0, 1.25, 1.5, 1.75, 2.0],
                'coef0':[2, 5, 9],
                }

            clf = GridSearchCV(SVR(), params, cv=3,
                   scoring='roc_auc',n_jobs=1)
            self.model =clf.fit(self.X_std , y)
            #print self.model.predict_proba(self.X_test)
        elif (modelType=='KRNCV_RG'):
            classifier_pipeline=None
            #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            #     'C': [1, 10, 100, 1000]},
            #    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            params={'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
                #'C':[1, 5, 10],
                'degree':[2,3],
                'gamma':[0.025, 1.0, 1.25, 1.5, 1.75, 2.0],
                'coef0':[2, 5, 9],
                }

            clf = GridSearchCV(KernelRidge(), param_grid=params,
                   cv=3,
                   scoring='roc_auc')
            self.model =clf.fit(self.X_std , y)
        elif (modelType=='KRN_RG'):
            clf = KernelRidge()
            self.model =clf.fit(self.X_std , y)
        else:
            SyntaxError("Error in modelType = 'LRCV', 'LG', 'RF', 'SVM', 'SVMCV', 'NBB', 'NBG' , 'DT'; \nLogistic Regression, Logistic Regression with Cross Validation, \nRandom Forest or Support Vector Machine with CV, \n DT  is decision Tree ")

    def get_feature_ranking(self):
        if self.selector_type == "None":
            return [0.0] * self.num_features
        elif (self.selector_type in ["LOGR", "LOGRCV", "SVR", "SVC"]):
            return self.model.coef_.tolist()[0]
        elif (self.selector_type in ["LINR"]):
            return self.model.coef_.tolist()
        if (self.selector_type in ["SVRGD", "SVCCV"]):
            return self.model.best_estimator_.coef_.tolist()[0]
        elif (self.selector_type in ["RF", "RF_RG", "DT", "DT_RG"]):
            return self.model.feature_importances_
        elif (self.selector_type in ["RFGD_RG", "DTGD_RG"]):
            return self.model.best_estimator_.feature_importances_
        else:
            print("no such selector specified: %s" % self.selector_type)
            exit(1)



def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]

     def recurse(left, right, child, lineage=None):
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print(node)


def get_code(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node, n_tabs):
                if (threshold[node] != -2):
                        print(n_tabs * "\t" + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node], n_tabs + 1)
                        print(n_tabs * "\t" + "} else {")
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node], n_tabs + 1)
                        print(n_tabs * "\t" + "}")
                else:
                        print(n_tabs * "\t" + "return " + str(value[node]))

        recurse(left, right, threshold, features, 0, 0)


def read_relevant_rules(relevant_rules_file):
    with open(relevant_rules_file) as f:
        rules = defaultdict(list)
        for line in f.readlines():
            action_name = line.split(" ", maxsplit=1)[0]
            rules[action_name].append(line.replace("\n", ""))
    return rules


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--training-folder", type=str, required=True, help="path to training set files (must be *.csv, where last column is the class, also need relevant_rules file)")
    argparser.add_argument("--selector-type", type=str, required=True, help="the type of the learning model: can be one of 'LRCV', 'LG', 'RF' , 'SVMCV','NBB', 'NBG', 'DT'")
    argparser.add_argument("--keep-duplicate-features", action="store_true", required=False, help="elimination and aggregation of duplicate feature vectors, default is eliminate")
    argparser.add_argument("--mean-over-duplicates", action="store_true", required=False, help="aggregating eliminated duplicate feature vectors by taking max or mean (default is max)")

    args = argparser.parse_args()

    # copy relevant_rules files to model_folder
    relevant_rules_file = os.path.join(args.training_folder, "relevant_rules")
    if (not os.path.isfile(relevant_rules_file)):
        print("WARNING: no \"relevant_rules\" file in training folder")
        sys.exit(1)

    print(f"Filter relevant rules based on feature selector {args.selector_type}")

    relevant_rules = read_relevant_rules(relevant_rules_file)
    useful_rules = defaultdict(list)

    usefulness = defaultdict(list)

    for file in os.listdir(args.training_folder):
        curr_file = os.path.join(args.training_folder, file)
        if (os.path.isfile(curr_file) and (file.endswith(".csv.bz2") or file.endswith(".csv"))):
            action_schema = file[:-8] if file.endswith(".csv.bz2") else file[:-4]
            print("handling action schema %s" % action_schema)

            dataset = helpers.get_dataset_from_csv(curr_file, args.keep_duplicate_features, not args.mean_over_duplicates)

            selector = FeatureSelector(dataset, args.selector_type)

            usefulness[action_schema] = sorted([(abs(rank), i) for i, rank in enumerate(selector.get_feature_ranking())])
            usefulness[action_schema].reverse()

            max_eval = usefulness[action_schema][0][0]
            fifth_eval = usefulness[action_schema][4][0] if len(usefulness[action_schema]) >= 5 else usefulness[action_schema][-1][0]
            num_useful_rules = 0

            for rank, i in usefulness[action_schema]:
                if (rank >= 0.01 and (rank > max_eval / 10 or num_useful_rules < 5) and num_useful_rules < 50):# or rank > fifth_eval / 2)):
                    useful_rules[action_schema].append(relevant_rules[action_schema][i])
                    num_useful_rules += 1
                    print(f"{round(rank, 2)} \t {relevant_rules[action_schema][i]} is useful")

            print()

    usefulness_file = os.path.join(args.training_folder, "rule_usefulness_" + args.selector_type.lower())
    write = True
    if (os.path.isfile(usefulness_file)):
        write = False
        response = input("WARNING: file already exists: %s override? Y/n " % usefulness_file)
        if (response in ["yes", "Yes", "YES", "y", "Y", ""]):
            write = True
    if (write):
        with open(usefulness_file, "w") as f:
            for schema in usefulness:
                for rank, i in usefulness[schema]:
                    f.write(str(round(rank, 4)) + "\t" + relevant_rules[schema][i] + "\n")

    useful_rules_file = os.path.join(args.training_folder, "useful_rules_" + args.selector_type.lower())
    write = True
    if (os.path.isfile(useful_rules_file)):
        write = False
        response = input("WARNING: file already exists: %s override? Y/n " % useful_rules_file)
        if (response in ["yes", "Yes", "YES", "y", "Y", ""]):
            write = True
    if (write):
        with open(useful_rules_file, "w") as f:
            for schema in useful_rules:
                for rule in useful_rules[schema]:
                    f.write(rule + "\n")





if __name__ == "__main__":
    main()
