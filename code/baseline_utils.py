import yaml
import eval_utils
import numpy as np

import sys
sys.path.append('../baseline')
import tree_explainer
from trustee import ClassificationTrustee, RegressionTrustee, ClassificationTrusteeSeq, RegressionTrusteeSeq

sys.path.append('../code')
import AE


# interface for DT-related baseline:
# 1. dt_sa: decision tree with randomly sampled anomalies
# 2. dt_trustee: DT with trustee sampling (iteratively from iid)
class DTUsage:
    def __init__(self, 
            nn_model,
            ad_thres,
            random_state = 0,
            verbose=True,
            is_trustee=False,
        ):
        self.random_state = random_state
        self.ad_thres = ad_thres
        self.verbose = verbose
        self.nn_model = nn_model
        self.min_impurity_decrease = None
        self.min_samples_leaf = None
        self.sample_min = 0.
        self.sample_max = 1.
        self.sample_rate = 1.
        self.dt = None # DT Tree

        self.is_trustee = is_trustee
        if self.is_trustee:
            self.top_k = None
            self.num_iter = 50
            self.num_stability_iter = 10
            self.samples_size = 0.3
            

    def load_config(self, config_pth):
        with open(config_pth, 'r') as file:
            configs = yaml.safe_load(file)
        self.min_impurity_decrease = configs['min_impurity_decrease']
        self.min_samples_leaf = configs['min_samples_leaf']
        self.sample_min = configs['sample_min']
        self.sample_max = configs['sample_max']
        self.sample_rate = configs['sample_rate']
        self.random_state = configs['random_state']
        if self.is_trustee: 
            self.top_k = configs['top_k']
            self.num_iter = configs['num_iter']
            self.num_stability_iter = configs['num_stability_iter']
            self.samples_size = configs['samples_size']
        
        
    def build_tree(self, X_train,):
        dh = tree_explainer.DatasetHandler(X_train, self.nn_model, self.ad_thres, verbose=self.verbose)
        X, y = dh.get_random_sampling_anomaly_dataset(sample_rate=self.sample_rate, 
                                value_range = (self.sample_min, self.sample_max))

        if self.is_trustee:
            trustee = ClassificationTrustee(expert=self.nn_model, ad_thres=self.ad_thres, random_state=self.random_state)
            trustee.fit(X, y, num_iter=self.num_iter, num_stability_iter=self.num_stability_iter, 
                        samples_size=self.samples_size, verbose=False, 
                        min_impurity_decrease = self.min_impurity_decrease,
                        min_samples_leaf = self.min_samples_leaf,
                        random_state = self.random_state,
                        top_k=self.top_k)
            self.dt, _, _, _ = trustee.explain()
        else:
            self.dt = tree_explainer.DecTreeExplainer(
                min_impurity_decrease = self.min_impurity_decrease,
                min_samples_leaf = self.min_samples_leaf,
                random_state = self.random_state,
                )

            self.dt.build(X, y)

    def get_n_leaves(self):
        if self.dt is None:
            raise RuntimeError('DT should be built first!')
        if self.is_trustee:
            return self.dt.tree_.n_leaves
        else:
            return self.dt.explainer.tree_.n_leaves
    
    def eval_fidelity(self, X_test):
        if self.is_trustee:
            tree_pred = self.dt.predict(X_test)
        else:
            tree_pred = self.dt.get_decision(X_test)
        nn_pred, _  = AE.test(self.nn_model, thres=self.ad_thres, X_test = X_test)
        f_acc = eval_utils.fidelity_acc(tree_pred,nn_pred)
        p_con, n_con = eval_utils.consistency(tree_pred,nn_pred)
        p_cre, n_cre = eval_utils.credibility(tree_pred,nn_pred)
        return f_acc, p_con, n_con, p_cre, n_cre

class DTSeqUsage(DTUsage):
    def build_tree(self, X_train_in, X_train_out):

        dh = tree_explainer.DatasetHandlerSeq(X_train_in, X_train_out, self.nn_model, self.ad_thres, verbose=True)
        X_in, X_out, y = dh.get_random_sampling_anomaly_dataset(sample_rate=self.sample_rate)

        if self.is_trustee:
            X = np.concatenate((X_in, np.reshape(X_out, (-1, 1))), axis=1)
            trustee = ClassificationTrusteeSeq(expert=self.nn_model, ad_thres=self.ad_thres, random_state=self.random_state)
            trustee.fit(X, y, num_iter=self.num_iter, num_stability_iter=self.num_stability_iter, 
                        samples_size=self.samples_size, verbose=False, 
                        min_impurity_decrease = self.min_impurity_decrease,
                        min_samples_leaf = self.min_samples_leaf,
                        random_state = self.random_state,
                        top_k=self.top_k)
            self.dt, _, _, _ = trustee.explain()
        else:
            self.dt = tree_explainer.DecTreeExplainerSeq(
                min_impurity_decrease = self.min_impurity_decrease,
                min_samples_leaf = self.min_samples_leaf,
                random_state = self.random_state,
            )

            self.dt.build(X_in, X_out, y)
    
    def eval_fidelity(self, X_valid_in, X_valid_out):
        if self.is_trustee:
            X_valid = np.concatenate((X_valid_in, np.reshape(X_valid_out, (-1, 1))), axis=1)
            tree_pred = self.dt.predict(X_valid)
        else:
            tree_pred = self.dt.get_decision(X_valid_in, X_valid_out)
        nn_pred, _  = self.nn_model.test_sequence_thres(X_valid_in, X_valid_out, ad_thres=self.ad_thres)
        f_acc = eval_utils.fidelity_acc(tree_pred,nn_pred)
        p_con, n_con = eval_utils.consistency(tree_pred,nn_pred)
        p_cre, n_cre = eval_utils.credibility(tree_pred,nn_pred)
        return f_acc, p_con, n_con, p_cre, n_cre


# interface for RT-related baseline:
# 1. rte: regression tree without LC 
# 2. rt with trustee sampling (iteratively from iid)
class RTUsage:
    def __init__(self, 
            nn_model,
            ad_thres,
            random_state = 0,
            verbose=True,
            is_trustee=False,
        ):
        self.random_state = random_state
        self.ad_thres = ad_thres
        self.verbose = verbose
        self.nn_model = nn_model
        self.min_impurity_decrease = None
        self.min_samples_leaf = None
        
        self.rt = None # RT Tree

        self.is_trustee = is_trustee
        if self.is_trustee:
            self.top_k = None
            self.num_iter = 50
            self.num_stability_iter = 10
            self.samples_size = 0.3
            

    def load_config(self, config_pth):
        with open(config_pth, 'r') as file:
            configs = yaml.safe_load(file)
        self.min_impurity_decrease = configs['min_impurity_decrease']
        self.min_samples_leaf = configs['min_samples_leaf']
        self.random_state = configs['random_state']
        if self.is_trustee: 
            self.top_k = configs['top_k']
            self.num_iter = configs['num_iter']
            self.num_stability_iter = configs['num_stability_iter']
            self.samples_size = configs['samples_size']
        
        
    def build_tree(self, X_train,):

        if self.is_trustee:
            trustee = RegressionTrustee(expert=self.nn_model, 
                    ad_thres=self.ad_thres, 
                    random_state=42)
            
            _, y = AE.test(self.nn_model, X_train, self.ad_thres)

            trustee.fit(X_train, y, num_iter=self.num_iter, num_stability_iter=self.num_stability_iter, 
                        samples_size=self.samples_size, verbose=False, 
                        min_impurity_decrease = self.min_impurity_decrease,
                        min_samples_leaf = self.min_samples_leaf,
                        random_state = self.random_state,
                        top_k=self.top_k)
            self.rt, _, _, _ = trustee.explain()
            
        else:
            self.rt = tree_explainer.RegTreeExplainer(self.nn_model, self.ad_thres,
                min_impurity_decrease = self.min_impurity_decrease,
                min_samples_leaf = self.min_samples_leaf,
                random_state = self.random_state,
            )

            self.rt.build(X_train)


    def get_n_leaves(self):
        if self.rt is None:
            raise RuntimeError('RT should be built first!')
        if self.is_trustee:
            return self.rt.tree_.n_leaves
        else:
            return self.rt.explainer.tree_.n_leaves
    
    def eval_fidelity(self, X_test):
        if self.is_trustee:
            tree_prob = self.rt.predict(X_test)
            tree_pred = np.zeros_like(tree_prob) # avoid replace error in the next two lines
            tree_pred[tree_prob<self.ad_thres] = 0
            tree_pred[tree_prob>=self.ad_thres] = 1
        else:
            tree_pred = self.rt.get_decision(X_test)
        nn_pred, _  = AE.test(self.nn_model, thres=self.ad_thres, X_test = X_test)
        f_acc = eval_utils.fidelity_acc(tree_pred,nn_pred)
        p_con, n_con = eval_utils.consistency(tree_pred,nn_pred)
        p_cre, n_cre = eval_utils.credibility(tree_pred,nn_pred)
        return f_acc, p_con, n_con, p_cre, n_cre


class RTSeqUsage(RTUsage):

    def build_tree(self, X_train_in, X_train_out):
        if self.is_trustee:
            trustee = RegressionTrusteeSeq(expert=self.nn_model, ad_thres=self.ad_thres, random_state=None)
            _, y = self.nn_model.test_sequence_thres(X_train_in, X_train_out, ad_thres=self.ad_thres)
            X_train = np.concatenate((X_train_in, np.reshape(X_train_out, (-1, 1))), axis=1)

            trustee.fit(X_train, y, num_iter=self.num_iter, num_stability_iter=self.num_stability_iter, 
                        samples_size=self.samples_size, verbose=False, 
                        min_impurity_decrease = self.min_impurity_decrease,
                        min_samples_leaf = self.min_samples_leaf,
                        random_state = self.random_state,
                        top_k=self.top_k)
            self.rt, _, _, _ = trustee.explain()
        else:
            self.rt = tree_explainer.RegTreeExplainerSeq(self.nn_model, ad_thres=self.ad_thres,
                min_impurity_decrease = self.min_impurity_decrease,
                min_samples_leaf = self.min_samples_leaf,
                random_state = self.random_state,
            )

            self.rt.build(X_train_in, X_train_out)
    
    def eval_fidelity(self, X_valid_in, X_valid_out):
        if self.is_trustee:
            X_valid = np.concatenate((X_valid_in, np.reshape(X_valid_out, (-1, 1))), axis=1)
            tree_prob = self.rt.predict(X_valid)
            tree_pred = np.zeros_like(tree_prob) # avoid replace error in the next two lines
            tree_pred[tree_prob<self.ad_thres] = 0
            tree_pred[tree_prob>=self.ad_thres] = 1
        else:
            tree_pred = self.rt.get_decision(X_valid_in, X_valid_out)
        nn_pred, _  = self.nn_model.test_sequence_thres(X_valid_in, X_valid_out, ad_thres=self.ad_thres)
        f_acc = eval_utils.fidelity_acc(tree_pred,nn_pred)
        p_con, n_con = eval_utils.consistency(tree_pred,nn_pred)
        p_cre, n_cre = eval_utils.credibility(tree_pred,nn_pred)
        return f_acc, p_con, n_con, p_cre, n_cre