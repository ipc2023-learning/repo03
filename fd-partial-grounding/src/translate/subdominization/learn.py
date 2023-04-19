#!/usr/bin/python


from sklearn.base import BaseEstimator

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier


class ConstantEstimator(BaseEstimator):
    def __init__(self, value):
        self.value = value

    def predict(self, actions):
        return [self.value] * len(actions)


class LearnRules:
    def __init__(self):
        pass



