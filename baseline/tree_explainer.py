'''
Reimplementation of Basic Tree-based Baselines for Explaining Anomaly Detection
'''

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn import tree
import time
import math
import copy
import sys
from tqdm import tqdm
import abc
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.append('../code/')
import AE
import tree_utils
from DeepLog import DeepLogAD
from general_utils import seq2tab, tab2seq


# base class of Tree-based Explainer
class TreeExplainerBase(abc.ABC):
    def __init__(self,
                #  nn_model, # DNN to be explained
                 verbose=False,
                 debug=False,
                 **tree_params, # tree params see sklearn.tree.DecisionTreeRegressor
                 ):
        
        # general params
        # self.nn_model = nn_model
        self.verbose = verbose
        self.debug = debug

        # regtree params
        self.tree_params = tree_params
        if 'min_impurity' in tree_params: # pruned by min_impurity (this is a customized params used by Trustee, not in original sklearn Tree params)
            self.pruned_min_impurity = tree_params['min_impurity']
            del tree_params['min_impurity']
        else:
            self.pruned_min_impurity = None

        if verbose:
            print('Decision Tree Params:', self.tree_params)
        
        self.explainer_class = None

    def build(self,
              features, # Tabular feature for training the explainer (model_input) 
              targets, # DNN outputs of features, need to be processed as 0/1 before for DecTreeExplainer
              ):
        self.explainer = self.explainer_class.fit(features, targets)
        if self.pruned_min_impurity is not None: 
            self.explainer = tree_utils.min_impurity_prune(self.explainer, self.pruned_min_impurity) # customized 'pre'- pruning params, simulate by post prune
    
    def get_decision(self, X_test): # 0/1 results of X_test
        pred_results = self.explainer.predict(X_test)
        if isinstance(self.explainer_class, DecisionTreeRegressor):
            _pred_results = np.zeros_like(pred_results) # avoid replace error in the next two lines
            _pred_results[pred_results<self.value_thres] = 0
            _pred_results[pred_results>=self.value_thres] = 1
            pred_results = _pred_results
        return pred_results
    

class DecTreeExplainer(TreeExplainerBase):
    def __init__(self, 
                 verbose=False,
                 debug=False,
                 **tree_params,):

        super(DecTreeExplainer, self).__init__(verbose, debug, **tree_params)
        self.explainer_class = DecisionTreeClassifier(**self.tree_params)

class RegTreeExplainer(TreeExplainerBase):
    def __init__(self, 
                 nn_model, # dnn model to be explained
                 value_thres, # threshold for anomaly detection used by tree_explainer
                 ad_thres=None, # threshold for anomaly detection used by nn_model (default None = same as value_thres)
                 verbose=False,
                 debug=False,
                 **tree_params,):

        super(RegTreeExplainer, self).__init__(verbose, debug, **tree_params)
        self.explainer_class = DecisionTreeRegressor(**self.tree_params)
        self.nn_model = nn_model
        self.value_thres = value_thres
        self.ad_thres = ad_thres
        if self.ad_thres is None:
            self.ad_thres = self.value_thres

    ## reload build for regression tree as the target (y_train) of X_train is given by nn_model
    def build(self, 
              features, # typically, features should be X_train_ben
              ):
        
        _, nn_prob = AE.test(self.nn_model, thres=self.ad_thres, X_test = features) # get anomaly score from nn model
        self.explainer = self.explainer_class.fit(features, nn_prob)
        if self.pruned_min_impurity is not None: 
            self.explainer = tree_utils.min_impurity_prune(self.explainer, self.pruned_min_impurity) # customized 'pre'- pruning params, simulate by post prune

# Get dataset for training anomaly detection explainers with ONLY normal data
class DatasetHandler:
    def __init__(self,
                 X_train_ben, # given a only normal training features/inputs
                 nn_model, # DNN to be explained
                 ad_thres, # threshold for anomaly detection used by nn_model
                 verbose=False,
                 debug=False,
                 ):

        self.X_train_ben = X_train_ben
        self.nn_model = nn_model
        self.ad_thres = ad_thres
        self.verbose = verbose
        self.debug = debug

    # set a threshold to treat e.g. 1% o f normal samples as anomalies for training
    def get_binarized_dataset(self, X_train):
        y_train, _ = AE.test(self.nn_model, thres=self.ad_thres, X_test = X_train)
        if self.verbose:
            print(f'Binarize normal data for anomaly detection explainers: NORMAL: {len(y_train[y_train==0])} ABNORMAL:{len(y_train[y_train==1])}')
        return X_train, y_train

    def get_random_sampling_anomaly_dataset(self, 
            sample_rate=1., # sampling ratio to the self.X_train_ben
            sample_method='uniform', # sampling method function
            value_range = (0., 1.), # sampling value rangle
            random_state=42, # fix sampling randomness
        ):
        if random_state is not None:
            np.random.seed(random_state)
        ben_shape = self.X_train_ben.shape
        if sample_method == 'uniform':
            X_samples =np.random.uniform(value_range[0], value_range[1], size=(int(ben_shape[0]*sample_rate),ben_shape[1]))
        else:
            raise NotImplementedError(f'select sample method {sample_method} is not supported!')
        
        X_concat, y_concat = self.get_binarized_dataset(np.concatenate((self.X_train_ben, X_samples)))

        if self.verbose:
            print(f'Randomly Sampling Anomalies with {sample_method} method and {sample_rate} ratio:')
            _y, _ = AE.test(self.nn_model, thres=self.ad_thres, X_test = self.X_train_ben)
            print(f"  - Original    Dataset: NORMAL: {len(_y[_y==0])}, ABNORMAL: {len(_y[_y==1])}")
            print(f"  - Concatenate Dataset: NORMAL: {len(y_concat[y_concat==0])}, ABNORMAL: {len(y_concat[y_concat==1])}")

        return  X_concat, y_concat  # X_samples, y_samples #
    

##############################################################################
############### For DeepLog-like Sequencial AD models ######################
##############################################################################

class DatasetHandlerSeq:
    def __init__(self,
                 X_train_in,
                 X_train_out,
                #  X_train_ben, # given a only normal training features/inputs
                #  nn_model, # DNN to be explained
                 deeplogad:DeepLogAD, # DNN model to be explained
                 ad_thres, # threshold for anomaly detection used by nn_model
                 verbose=False,
                 debug=False,
                 ):

        self.X_train_in, self.X_train_out = X_train_in, X_train_out
        self.nn_ad = deeplogad
        self.ad_thres = ad_thres
        self.verbose = verbose
        self.debug = debug

    # set a threshold to treat e.g. 1% o f normal samples as anomalies for training
    def get_binarized_dataset(self, X_train):
        X_train_in, X_train_out = tab2seq(X_train, self.nn_ad.window_size)
        y_train, _ = self.nn_ad.test_sequence_thres(X_train_in, X_train_out, ad_thres=self.ad_thres)
        if self.verbose:
            print(f'Binarize normal data for anomaly detection explainers: NORMAL: {len(y_train[y_train==0])} ABNORMAL:{len(y_train[y_train==1])}')
        return X_train, y_train

    def get_random_sampling_anomaly_dataset(self, 
            sample_rate=1., # sampling ratio to the self.X_train_ben
            random_state=42, # fix sampling randomness
        ):
        if random_state is not None:
            np.random.seed(random_state)

        sample_size = int(len(self.X_train_in)*sample_rate)
        X_samples = np.random.randint(0, 28, size=(sample_size, 6))
        
        X_train_ben = seq2tab(self.X_train_in, self.X_train_out)
        X_concat, y_concat = self.get_binarized_dataset(np.concatenate((X_train_ben, X_samples)))

        if self.verbose:
            print(f'Randomly Sampling Anomalies for SEQ case with {sample_rate} ratio:')
            _y, _ = self.nn_ad.test_sequence_thres(self.X_train_in, self.X_train_out, self.nn_ad.window_size)
            print(f"  - Original    Dataset: NORMAL: {len(_y[_y==0])}, ABNORMAL: {len(_y[_y==1])}")
            print(f"  - Concatenate Dataset: NORMAL: {len(y_concat[y_concat==0])}, ABNORMAL: {len(y_concat[y_concat==1])}")

        X_concat_in, X_concat_out = tab2seq(X_concat, self.nn_ad.window_size)
        return X_concat_in, X_concat_out, y_concat # X_concat, y_concat  
    
# base class of Tree-based Explainer
class TreeExplainerBaseSeq(TreeExplainerBase):
    def __init__(self,
                 verbose=False,
                 debug=False,
                 **tree_params, # tree params see sklearn.tree.DecisionTreeRegressor
                 ):
        
        super(TreeExplainerBaseSeq, self).__init__(verbose, debug, **tree_params)

    def build(self,
              X_in,  
              X_out, 
              y, # DNN outputs of features, need to be processed as 0/1 before for DecTreeExplainer
              ):
        features = seq2tab(X_in, X_out)
        self.explainer = self.explainer_class.fit(features, y)
        if self.pruned_min_impurity is not None: 
            self.explainer = tree_utils.min_impurity_prune(self.explainer, self.pruned_min_impurity) # customized 'pre'- pruning params, simulate by post prune
    
    def get_decision(self, X_test_in, X_test_out): # 0/1 results of X_test
        X_test = seq2tab(X_test_in, X_test_out)
        return super().get_decision(X_test)

class RegTreeExplainerSeq(TreeExplainerBaseSeq):
    def __init__(self, 
                 nn_ad, # dnn model to be explained
                 ad_thres,
                 verbose=False,
                 debug=False,
                 **tree_params,):

        super(RegTreeExplainerSeq, self).__init__(verbose, debug, **tree_params)
        self.explainer_class = DecisionTreeRegressor(**self.tree_params)
        self.nn_ad = nn_ad
        self.ad_thres = ad_thres
        self.value_thres = ad_thres

    ## reload build for regression tree as the target (y_train) of X_train is given by nn_model
    def build(self, 
              X_in,  
              X_out, 
              ):
        
        _, nn_prob = self.nn_ad.test_sequence(X_in, X_out)
        features = seq2tab(X_in, X_out)
        self.explainer = self.explainer_class.fit(features, nn_prob)
        if self.pruned_min_impurity is not None: 
            self.explainer = tree_utils.min_impurity_prune(self.explainer, self.pruned_min_impurity) # customized 'pre'- pruning params, simulate by post prune
        

class DecTreeExplainerSeq(TreeExplainerBaseSeq):
    def __init__(self, 
                 verbose=False,
                 debug=False,
                 **tree_params,):

        super(DecTreeExplainerSeq, self).__init__(verbose, debug, **tree_params)
        self.explainer_class = DecisionTreeClassifier(**self.tree_params)    