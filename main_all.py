'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np

from data_loader import data_loader, data_loader2
from gain_ori import gain
from egain import egain
from utils import rmse_loss, normalization
from missingpy import MissForest
from sklearn import metrics
from math import sqrt
from impyute.imputation.cs import mice
import pandas as pd
from autoimpute.imputations import MiceImputer, SingleImputer, MultipleImputer
from autoimpute.analysis import MiLinearRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')

def auc_dt(impute,data):
    df1 = pd.read_csv("/content/tommy/data/{}.csv".format(data))
    df1 = df1.rename(columns={'target': 'Class'})
    df1['Class'] = pd.factorize(df1['Class'])[0] + 1
    col_Names=list(df1.columns)
    df = pd.read_csv("/content/tommy/data/{}.csv".format(impute), names=col_Names)
    X = df.drop(['Class'], axis=1)
    targets = df1['Class'].values

    X_train, test_x, y_train, test_lab = train_test_split(X,targets,test_size = 0.3,random_state = 42)
    clf = DecisionTreeClassifier( random_state = 42) # max_depth =3,
    clf.fit(X_train, y_train)
    test_pred_decision_tree = clf.predict(test_x)
    return  metrics.accuracy_score(test_lab, test_pred_decision_tree)

    #scf = StratifiedShuffleSplit(n_splits=5)
    #score_dct = cross_val_score(DecisionTreeClassifier(max_depth=5), X, targets, cv=scf, scoring='accuracy')
    #metrics.accuracy_score(test_lab, test_pred_decision_tree)

    #print('Method: {}'.format(impute))
    #print('Mean Validation AUC AUC: {}'.format(round(np.mean(score_dct),6)))

def auc_mlp(impute,data):
    df1 = pd.read_csv("/content/tommy/data/{}.csv".format(data))
    df1 = df1.rename(columns={'target': 'Class'})
    df1['Class'] = pd.factorize(df1['Class'])[0] + 1
    col_Names=list(df1.columns)
    df = pd.read_csv("/content/tommy/data/{}.csv".format(impute), names=col_Names)
    X = df.drop(['Class'], axis=1)
    targets = df1['Class'].values

    X_train, test_x, y_train, test_lab = train_test_split(X,targets,test_size = 0.3,random_state = 42)
    #clf = DecisionTreeClassifier( random_state = 42) # max_depth =3,
    #clf = MLPClassifier(hidden_layer_sizes= X_train.shape[1]//2,  early_stopping=True) #max_iter=500,
    clf = MLPClassifier(hidden_layer_sizes=X_train.shape[1]//2, max_iter=500, early_stopping=True, 
      #learning_rate_init=0.01,
      learning_rate='constant')
    clf.fit(X_train, y_train)
    test_pred_decision_tree = clf.predict(test_x)
    return  metrics.accuracy_score(test_lab, test_pred_decision_tree)


def auc_lr(impute,data):
    df1 = pd.read_csv("/content/tommy/data/{}.csv".format(data))
    df1 = df1.rename(columns={'target': 'Class'})
    df1['Class'] = pd.factorize(df1['Class'])[0] + 1
    col_Names=list(df1.columns)
    df = pd.read_csv("/content/tommy/data/{}.csv".format(impute), names=col_Names)
    X = df.drop(['Class'], axis=1)
    targets = df1['Class'].values
    X_train, test_x, y_train, test_lab = train_test_split(X,targets,test_size = 0.3,random_state = 42)
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    test_pred_decision_tree = clf.predict(test_x)
    return  metrics.accuracy_score(test_lab, test_pred_decision_tree)



def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  random    = args.seed
  time      = args.time

  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'time': args.time}
   # Load data and introduce missingness
 
  #ori_data_x, miss_data_x, data_m = data_loader2(data_name, miss_rate,random)
  

  gan_rs, egain_rs, mice_rs,miss_rs, miss_lr, mice_lr, miss_mlp, mice_mlp = [],[],[],[],[],[],[],[];

  for i in range(time):

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader2(data_name, miss_rate,time)
    if i % 5 == 0:
        print('=== Working on {}/{} ==='.format(i, time))

    # Impute missing data
    imputed_data_x = gain(miss_data_x, gain_parameters)
    imputed_data_x_e = egain(miss_data_x, gain_parameters)

    imp_mf = IterativeImputer(estimator = DecisionTreeRegressor(), max_iter = 2)
    imputed_data_mf = imp_mf.fit_transform(miss_data_x)
    imputed_data_mf, _ = normalization(imputed_data_mf)

    imp_mice = IterativeImputer(max_iter = 2)
    imputed_data_mice = imp_mice.fit_transform(miss_data_x)
    imputed_data_mice, _ = normalization(imputed_data_mice)

    
    # Report the RMSE performance
    rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
    rmse_e = rmse_loss (ori_data_x, imputed_data_x_e, data_m)
    rmse_mf = rmse_loss (ori_data_x, imputed_data_mf, data_m)
    rmse_mice = rmse_loss (ori_data_x, imputed_data_mice, data_m)

    
    mi_data = miss_data_x.astype(float)
    no, dim = imputed_data_x.shape
    miss_data = np.reshape(mi_data,(no,dim))
    np.savetxt("data/missing_data.csv",mi_data,delimiter=',',fmt='%1.2f')

    gan_rs.append(rmse)
    egain_rs.append(rmse_e)
    mice_rs.append(rmse_mice)
    miss_rs.append(rmse_mf)
    
    np.savetxt("data/imputed_data_gain.csv",imputed_data_x, delimiter=',',  fmt='%d')
    np.savetxt("data/imputed_data_egain.csv",imputed_data_x_e, delimiter=',',  fmt='%d')

  #print(gan_rs,egain_rs)
  print('RMSE  GAIN: {} ± {}'.format(round(np.mean(gan_rs)*1,2), round(np.std(gan_rs),2)))
  print('RMSE EGAIN: {} ± {}'.format(round(np.mean(egain_rs)*1,2), round(np.std(egain_rs),2)))
  print('RMSE  MICE: {} ± {}'.format(round(np.mean(mice_rs)*1,2), round(np.std(mice_rs),2)))
  print('RMSE MFORE: {} ± {}'.format(round(np.mean(miss_rs)*1,2), round(np.std(miss_rs),2)))
  
  # MissForest

  #print()
  #print('=== MissForest RMSE ===')
  #data = miss_data_x
  #imp_mean = MissForest(max_iter = 1)
  #miss_f = imp_mean.fit_transform(data)
  #miss_f = pd.DataFrame(imputed_train_df)
  #rmse_MF = rmse_loss (ori_data_x, miss_f, data_m)
  #print('RMSE Performance: ' + str(np.round(rmse_MF, 6)))
  #np.savetxt("data/imputed_data_MF.csv",miss_f, delimiter=',',  fmt='%d')
  #print( 'Save results in Imputed_data_MF.csv')

  # MICE From Auto Impute
  #print()
  #print('=== MICE of Auto Impute RMSE ===')
  #data_mice = pd.DataFrame(miss_data_x)
  #mi = MiceImputer(k=1, imp_kwgs=None, n=1, predictors='all', return_list=True,
  #      seed=None, strategy='interpolate', visit='default')
  #mice_out = mi.fit_transform(data_mice)
  #c = [list(x) for x in mice_out]
  #c1= c[0]
  #c2=c1[1]
  #c3=np.asarray(c2)
  #mice_x=c3
  #print('here :', mice_x, miss_f, miss_f.shape)
  #rmse_MICE = rmse_loss (ori_data_x, mice_x, data_m)
  #print('=== MICE of Auto Impute RMSE ===')
  #print('RMSE Performance: ' + str(np.round(rmse_MICE, 6)))
  #np.savetxt("data/imputed_data_MICE.csv",mice_x, delimiter=',',  fmt='%d')
  #print( 'Save results in Imputed_data_MICE.csv')


  return imputed_data_x, rmse


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam', 'breast', 'credit', 'news','blood','vowel','ecoli','ionosphere','parkinsons','seedst','vehicle','vertebral','wine','banknote','balance','yeast'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  parser.add_argument(
      '--time',
      help='number of repeated process',
      default=1,
      type=int)
  parser.add_argument(
      '--seed',
      help='number of repeated process',
      default=42,
      type=int)
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)
