'''
implementation the variant of GEAD for deeplog-like sequencial AD models
NOTE: we transform the negative score to anomaly score here
'''
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import copy
import sys
from tqdm import tqdm
from itertools import product
import json

from DeepLog import DeepLogAD
from gead import GEADBase
import tree_utils
import eval_utils
from general_utils import print_title, PRINT_LINEWIDTH, seq2tab, tab2seq
from general_utils import DEVICE

# torch.autograd.set_detect_anomaly(True)

MIN_LOSS = 0.01 # use for early stop of LC augmentation
PROB_SCALE = 1. # scaling probs for sake of illstrution
FAST_AUG_RATE = 0.1 # if a lc leaf has X_train more than FAST_AUG_RATE, will reduce aug_step

class GEADSeqBase(GEADBase):
    def __init__(self,
                 deeplogad:DeepLogAD, # DNN model to be explained
                 ad_thres=0.99, #NOTE: must set for deeplog
                 dt_lc_pctg=0.99, # DT anomaly percentage of LC region e.g. 0.95 means we identify lowest 5% as low confidence 
                 nn_lc_delta=0.01, #NOTE: for seq case, we identify NN_LC by **ad_thres - nn_lc_delta**
                 aug_step=50, # total steps for data augmentation
                 #  aug_pace=0.01, # perturb rate for data augmentation
                 aug_dim=5, # limited perturb dims for data augmentation
                 aug_upperbound=1.1, # when generating abnormal samples, limit the max value of grad ascent (as aug_upperbound*self.value_thres) 
                 aug_lowerbound=0.9, # when generating abnormal samples, limit the min value of grad ascent (as aug_lowerbound*self.value_thres)
                 subtree_type='dec', #tree type of lc subtrees ['dec','reg']
                 #  lc_ignore_bignode = False, # if True, lc for dt (impurity) will ignore the lc sample number > (1-dt_lc_pctg) (no need for value) 
                 dtlc_correct=True, # only valid when subtree_type='dec', if True, will correct inconsistent decision between rt-value and dt-value for some leaf
                 verbose=False,
                 debug=False,
                 **tree_params, # tree params see sklearn.tree.DecisionTreeRegressor/DecisionTreeClassifier
                 ):
        
        # general params
        self.nn_ad = deeplogad
        self.ad_thres = ad_thres
        self.prob_scale = PROB_SCALE
        self.value_thres = self.ad_thres * self.prob_scale
        self.verbose = verbose
        self.debug = debug

        # set feature_name:
        feature_names = [f'I_{i+1}' for i in range(self.nn_ad.window_size)] + ['OUT']
        self.feature_names = np.asarray(feature_names)

        # pre-defined tree params (for BOTH roottree and subtree)
        if 'min_impurity' in tree_params: # pruned by min_impurity (this is a customized params used by Trustee, not in original sklearn Tree params)
            self.pruned_min_impurity = tree_params['min_impurity']
            del tree_params["min_impurity"] # because sklearn DT/RT do NOT support this params
        else:
            self.pruned_min_impurity = None
        self.tree_params = tree_params

        # low confidence identification params
        self.dt_lc_pctg = dt_lc_pctg
        self.nn_lc_delta = nn_lc_delta

        # LC tree params
        assert subtree_type in ['dec', 'reg']
        self.subtree_type = subtree_type
        if self.subtree_type == 'dec':
            self.dtlc_correct = dtlc_correct 

        # params for data augmentation 
        self.aug_step = aug_step
        self.aug_dim = aug_dim
        self.aug_upperbound = aug_upperbound # when generating abnormal samples, limit the max value of grad ascent (as max_bound_rate*self.value_thres) 
        self.aug_lowerbound = aug_lowerbound  # when generating abnormal samples, limit the max value of grad ascent (as max_bound_rate*self.value_thres)

    # used for change member params externally
    # use cases: grid_search()
    def set_params(self, **params):
        for key, value in params.items():
            # if not hasattr(self, key):
            #     raise RuntimeWarning(f'Parameter {key} is not created! must created before calling `set_params()`')
            # else:
            setattr(self, key, value)
    
    def show_tree(self, tree, feature_names=None, nodebook=True):
        if feature_names is None:
            feature_names = self.feature_names
        n_leaves = tree.tree_.n_leaves
        print('n_leaves', n_leaves)
        if n_leaves < 20:
            plt.figure(figsize=(15,10))
        elif n_leaves < 100:
            plt.figure(figsize=(30,20))
        else:
            plt.figure(figsize=(50,35))
        tree.plot_tree(tree, feature_names=feature_names, filled=True)
        if not nodebook:
            save_name = 'tmp/tree_'+str(round(time.time()))+'.pdf'
            plt.savefig(save_name)
            print('Save fig at', save_name)
    
    ## generate the root regression tree
    def get_roottree(self, 
                     X_train:np.ndarray, # training feat of NN/DT (NOTE: all benign)
                    ):
        
        self.value_thres *= self.prob_scale
        X_train_in, X_train_out = tab2seq(X_train, self.nn_ad.window_size)
        self.nn_pred_train, nn_prob_train = self.nn_ad.test_sequence_thres(X_train_in, X_train_out, ad_thres=self.ad_thres)
        self.nn_prob_train = nn_prob_train * self.prob_scale
        regressor = tree.DecisionTreeRegressor(**self.tree_params)
        self.roottree = regressor.fit(X_train, self.nn_prob_train)
        if self.pruned_min_impurity is not None: 
            self.roottree = tree_utils.min_impurity_prune(self.roottree, self.pruned_min_impurity) # customized 'pre'- pruning params, simulate by post prune
        self.impurity_thres = tree_utils.get_impurity_thres(self.roottree, self.dt_lc_pctg)

        # NOTE: for seq case, we identify NN_LC by **value_thres - nn_lc_delta**
        nn_lc_thres = self.value_thres - self.nn_lc_delta
        self.lc_leave_indices, self.dt_lc_leave_indices, self.nn_lc_leave_indices = \
            tree_utils.get_LC_leaves(self.roottree, nn_lc_thres, self.impurity_thres, verbose=self.verbose)
        if self.verbose:
            print("Get root regression tree from scratch")
            print(" - Number of leaf nodes:", self.roottree.tree_.n_leaves)
            print(" - Max Depth:", self.roottree.tree_.max_depth)
            print(" - Prob Threshold:", self.value_thres)
            print(" - Impurity Threshold:", self.impurity_thres)        
            print(f" - Low-confidence leaves: {len(self.lc_leave_indices)} ({len(self.lc_leave_indices)/self.roottree.tree_.n_leaves:.2f})"
                  f"(dt_lc:{len(self.dt_lc_leave_indices)}, nn_lc:{len(self.nn_lc_leave_indices)})")
    
    ## generate data for each low confidence region/leaf
    def lc_data_augment(self, X_train, 
                        n_total = None, # if not None, will ignore self.aug_step, generate `n_total` samples for each LC
                        # random_pace = False, # if True, will set lr as random(0, self.aug_pace) instead of self.aug_pace
                        # all_dim = True, # if True, will perturb all features w/o. feature mask (NOTE: the depth of subtree is limited by subtree_params in this case)
                        # fix_dim = False, # if True, total modificated dims is fixed to top-k (k=self.aug_dim) grad at the first epoch 
                        ):
        '''
        prepare related data in each low confidence region/leaf (self.lc_leave_data)
        element of self.lc_leave_data:
            {'decision_mask':[a list of decision path], 'X_train_indices': [a list of X_train_idx falls into this leaf], 'value':, 'impurity':}
        '''
        self.lc_leave_data = [] # length and order is same as self.lc_leave_indices
        X_leaf_indices = self.roottree.apply(X_train)
        tree_feature = self.roottree.tree_.feature
        tree_value = self.roottree.tree_.value.reshape(-1)
        tree_impurity = self.roottree.tree_.impurity.reshape(-1)
        # if self.lc_ignore_bignode: filtered_lc_leave_indice = copy.copy(self.lc_leave_indices)
        for lc_idx in self.lc_leave_indices:
            X_train_indices = np.where(X_leaf_indices==lc_idx)[0]
            # if self.lc_ignore_bignode and len(X_train_indices) >= (1-self.dt_lc_pctg)*len(X_train):
            #     filtered_lc_leave_indice.remove(lc_idx)
            #     continue
            lcd = {}
            lcd['X_train_indices'] = X_train_indices
            decision_matrix = self.roottree.decision_path(X_train[lcd['X_train_indices'][0:1]])
            decision_path = decision_matrix.indices[decision_matrix.indptr[0] : decision_matrix.indptr[1]]
            lcd['decision_mask'] = tree_feature[decision_path[:-1]] # :-1 because leaf node doesn't have feature idx
            lcd['value'] = tree_value[lc_idx]
            lcd['impurity'] = tree_impurity[lc_idx]
            self.lc_leave_data.append(lcd)
            # if self.debug: print(len(lcd['X_train_indices'])) # print(self.feature_names[lcd['decision_mask']], len(lcd['X_train_indices']))
        # if self.lc_ignore_bignode:
        #     print(f'WARNING: Ignore {len(self.lc_leave_indices)-len(filtered_lc_leave_indice)} LC nodes with too many samples !')
        #     self.lc_leave_indices = filtered_lc_leave_indice
        #     print('RENEW LC Leaves Number:', len(self.lc_leave_indices))

        n_aug = 0 # number of samples after augmentation related to LC leaves
        n_lclf_xtrain = 0 # number of original training sample for all LC leaves
        # generate samples for each low confidence region/leaf
        if self.verbose: iter = tqdm(self.lc_leave_data, ncols=PRINT_LINEWIDTH)
        else: iter = self.lc_leave_data
        for i,lcd in enumerate(iter):
            if not self.verbose:
                print(f"(AUG:{i}/{len(iter)},N_sample:{len(lcd['X_train_indices'])})", end='\r')
            if self.debug: 
                print(f"{len(lcd['X_train_indices'])} samples, value is {lcd['value']} (thres:{self.value_thres}) ")
            # X_aug = self._perturb_augment(X_train, X_train[lcd['X_train_indices']], lcd['decision_mask'], lcd['value'], n_total, random_pace, all_dim, fix_dim)
            X_aug = self._perturb_augment_batch(X_train, X_train[lcd['X_train_indices']], lcd['decision_mask'], lcd['value'], n_total)
            lcd['X_aug'] = X_aug
            n_aug += len(X_aug)
            n_lclf_xtrain += len(lcd['X_train_indices'])
        
        if self.verbose: print(f'NOTICE: Finish data augmentataion! Total augment {n_lclf_xtrain} -> {n_aug} samples')
    
    def _perturb_augment_batch(self, X_train, X_lc, feature_mask, value, # these params are get from self.lc_leave_data
        n_total, # configurations, see comments in lc_data_augment
    ):  
    
        # get deeplog params
        window_size, device, num_classes, num_candidates = self.nn_ad.window_size, self.nn_ad.device, self.nn_ad.num_classes, self.nn_ad.num_candidates

        debug = self.debug
        ealy_stop_thres = MIN_LOSS # stop generation if grad are less than ealy_stop_thres
        max_bound_rate = self.aug_upperbound # when generating abnormal samples, limit the max value of grad ascent (as max_bound_rate*self.value_thres) 
        min_bound_rate = self.aug_lowerbound # when generating abnormal samples, limit the max value of grad ascent (as max_bound_rate*self.value_thres)

        fast_step = int(len(X_lc) * FAST_AUG_RATE)
        if FAST_AUG_RATE < 1. and fast_step>1:
            X_lc = X_lc[::fast_step]
        # X_aug = np.empty((0, X_lc.shape[1]))
        X_aug = np.copy(X_lc)

        X_lc_in, X_lc_out = tab2seq(X_lc, window_size)
        seq = torch.tensor(X_lc_in, dtype=torch.long).view(-1, window_size).to(device)
        seq = F.one_hot(seq,num_classes=num_classes).float()
        label = torch.tensor(X_lc_out).view(-1).to(device)
        Bound = nn.ReLU()

        # set aug_step for each training sample according to `self.aug_step` and `n_total`
        if n_total is not None:
            aug_step = max(1, n_total//(len(X_lc)*2)) # *2 because two directions
        else:
            aug_step = self.aug_step
        aug_step = min(num_classes, aug_step) # maxinum value of aug_step is num_classes (othervise will have duplicated perturbations)
        
        X_seq_lc = seq.clone().detach().requires_grad_(True)
        
        if window_size in feature_mask: #if cannot perturb output position, perturb X_in

            ## Generating abnormal LC samples
            perturb_thres = self.ad_thres*max_bound_rate
            if perturb_thres >= 1.: perturb_thres = self.ad_thres # set upper bound for ReLU
            for step in range(aug_step):
                output = self.nn_ad.model(X_seq_lc, device) # size: batch size, class size(28)
                
                output = 1. - F.softmax(output, dim=1) # covert to anomaly score
                row_indices = torch.arange(0, output.size(0)) 
                col_indices = torch.from_numpy(X_lc_out)
                
                bound_loss = -Bound(perturb_thres-output[row_indices, col_indices]).sum() # output should be increased (become abnormal)

                bound_loss.backward()
                X_in_grad = X_seq_lc.grad.clone()
                X_seq_lc.grad.zero_()
                in_mask = feature_mask[feature_mask != window_size] # delete output position mask to generate X_in mask
                with torch.no_grad():
                    X_in_grad[:, in_mask, :] = 0
                    max_v_dim2, max_idx_dim2 = torch.max(X_in_grad, dim=2) # shape: len(X_lc) * window_size
                    max_v_dim1, max_idx_dim1 = torch.max(max_v_dim2, dim=1) # shape: len(X_lc)
                    
                    for i in range(len(X_in_grad)):
                        _dim1_idx = max_idx_dim1[i] # X_in/grad size: batch, *window size*, class size(28)
                        _dim2_idx = max_idx_dim2[i, _dim1_idx] # X_in/grad size: batch, window size, *class size(28)*
                        if max_v_dim1[i] > 0.: # ensure the max_grad is not zero
                            
                            X_seq_lc[i, _dim1_idx, X_lc_in[i, _dim1_idx]] = 0.
                            X_seq_lc[i, _dim1_idx, _dim2_idx] = 1.
                            
                    X_lc_in_p = torch.argmax(X_seq_lc, dim=-1).numpy()
                    X_lc_p = seq2tab(X_lc_in_p, X_lc_out)
                    
                    X_aug = np.concatenate((X_aug, X_lc_p))
                    

            ## Generating Normal LC samples
            perturb_thres = self.ad_thres*min_bound_rate
            for step in range(aug_step):
                output = self.nn_ad.model(X_seq_lc, device) # size: batch size, class size(28)
                output = 1. - F.softmax(output, dim=1) # covert to anomaly score
                row_indices = torch.arange(0, output.size(0)) 
                col_indices = torch.from_numpy(X_lc_out)
                bound_loss = -Bound(output[row_indices, col_indices]-perturb_thres).sum() # output should be increased (become abnormal)
                bound_loss.backward()
                X_in_grad = X_seq_lc.grad.clone()
                X_seq_lc.grad.zero_()
                in_mask = feature_mask[feature_mask != window_size] # delete output position mask to generate X_in mask
                with torch.no_grad():
                    X_in_grad[:, in_mask, :] = 0
                    max_v_dim2, max_idx_dim2 = torch.max(X_in_grad, dim=2) # shape: len(X_lc) * window_size
                    max_v_dim1, max_idx_dim1 = torch.max(max_v_dim2, dim=1) # shape: len(X_lc)
                    for i in range(len(X_in_grad)):
                        _dim1_idx = max_idx_dim1[i] # X_in/grad size: batch, *window size*, class size(28)
                        _dim2_idx = max_idx_dim2[i, _dim1_idx] # X_in/grad size: batch, window size, *class size(28)*
                        if max_v_dim1[i] > 0.: # ensure the max_grad is not zero
                            X_seq_lc[i, _dim1_idx, X_lc_in[i, _dim1_idx]] = 0.
                            X_seq_lc[i, _dim1_idx, _dim2_idx] = 1.
                            
                    X_lc_in_p = torch.argmax(X_seq_lc, dim=-1).numpy()
                    X_lc_p = seq2tab(X_lc_in_p, X_lc_out)
                    
                    X_aug = np.concatenate((X_aug, X_lc_p))
                    

        else: # if can perturb output position, perturb it (X_out)
            output = self.nn_ad.model(X_seq_lc, device) # size: batch size, class size(28)
            output = 1. - F.softmax(output, dim=1) # covert to anomaly score
            X_lc_out_p_arr = [] # (batch_size, aug_size+1) each line is the all candidate `label`/X_out for sample i in batch 
            for i in range(len(X_lc_out)): # batch size
                max_indices = torch.argmax(output, dim=1).numpy().tolist()
                each_dir = max(1, aug_step//2) # equally augmented for positive and negative output
                pos_indices = max_indices[:each_dir]
                neg_indices = max_indices[-each_dir:]
                concat_indices = pos_indices+neg_indices+[X_lc_out[i]]
                X_lc_out_p_arr.append(concat_indices)
            X_lc_out_p_arr = np.asarray(X_lc_out_p_arr).T # (aug_size+1, batch_size) exchange row and col, now each line is all X_output with the length of batch_size
            for i in range(len(X_lc_out_p_arr)): 
                X_lc_p = seq2tab(X_lc_in, X_lc_out_p_arr[i])
                X_aug = np.concatenate((X_aug, X_lc_p))

        return X_aug

    ## get fine-grained rules (in form of trees) for all low confidence regions
    def get_lc_trees(self, X_train,  # note that, this X_train must same with the one in self.lc_data_augment()
                    **subtree_params, 
                    ): 
        # set subtree pre-defined params
        if not subtree_params:
            subtree_params = copy.copy(self.tree_params)
            subtree_params['max_depth'] = self.aug_dim

        self.lc_trees = [] # length and order is same as self.lc_leave_indices
        self.dtlc_leave_values = [] # values of leaves for each lc tree (only valid when lc tree is DT)
        if self.verbose: print(f'Generate fine-grained rules for LC Leaves: (no., leaves, max_depth):')
        n_aug_rules = 0
        for i, lcd in enumerate(self.lc_leave_data):
            if self.subtree_type == 'dec':
                lc_tree, leave_values = self._generate_lc_dectree(X_train[lcd['X_train_indices']], lcd['X_aug'], lcd['decision_mask'], subtree_params)
                self.dtlc_leave_values.append(leave_values)
            else:
                lc_tree = self._generate_lc_regtree(X_train[lcd['X_train_indices']], lcd['X_aug'], lcd['decision_mask'], subtree_params)
            self.lc_trees.append(lc_tree)
            if self.verbose:
                n_info_p = PRINT_LINEWIDTH//(1+3+1+3+1+2+1) # how many subtree info can be printed in each line
                print(f'[{i:>3},{lc_tree.tree_.n_leaves:>3},{lc_tree.tree_.max_depth:>2}]', end='')
                if i%n_info_p==(n_info_p-1) or i==len(self.lc_leave_indices)-1: print('')
            n_aug_rules += lc_tree.tree_.n_leaves

        if self.verbose:
            print(f'NOTICE: Finish fine-grained rules genereation! Total augment {n_aug_rules} rules')

    ## generate regtree for single lc leaf augment data
    def _generate_lc_regtree(self, X_train_lc, X_aug_lc, feature_mask, subtree_params):
        X_concat = np.concatenate((X_train_lc, X_aug_lc))

        X_in, X_out = tab2seq(X_concat, self.nn_ad.window_size)
        _, nn_prob = self.nn_ad.test_sequence(X_in, X_out)
        nn_prob *= self.prob_scale

        regressor = tree.DecisionTreeRegressor(**subtree_params)
        feature_indices = self._remain_feature_indices(feature_mask)
        lc_regtree = regressor.fit(X_concat[:,feature_indices], nn_prob)
        # if self.verbose: self.show_regtree(lc_regtree, feature_names=self.feature_names[feature_indices], nodebook=False)
        return lc_regtree
    
    ## generate dectree for single lc leaf augment data
    ## NOTE: different from _generate_lc_regtree, we need additionally get ** leave_values ** for whole-tree merging
    def _generate_lc_dectree(self, X_train_lc, X_aug_lc, feature_mask, subtree_params):
        X_concat = np.concatenate((X_train_lc, X_aug_lc))

        X_in, X_out = tab2seq(X_concat, self.nn_ad.window_size)
        nn_pred, nn_prob = self.nn_ad.test_sequence_thres(X_in, X_out, ad_thres=self.ad_thres) # nn_pred is used for training lc_dectree, nn_prob is used for leave_values
        nn_prob *= self.prob_scale

        feature_indices = self._remain_feature_indices(feature_mask)
        decisoner = tree.DecisionTreeClassifier(**subtree_params)
        lc_dectree = decisoner.fit(X_concat[:,feature_indices], nn_pred)
        if self.debug: print(f'nn_pred==0: {np.sum(nn_pred==0)}/{sum(nn_pred)}') # self.value_thres

        # get average sample value (by nn_pred) for each single leaf in lc_dectree
        # used for merged_tree (convert dt subtree value to rt subtree value)
        X_leaf_indices = lc_dectree.apply(X_concat[:,feature_indices])
        leave_values = {} # {leaf_idx (in lc_dectree): average sample value (by nn_prob) in this leaf}
        leave_indices = list(set(X_leaf_indices)) # all leave idx in this lc_dectree
        leave_indices.sort() 
        if len(leave_indices) != lc_dectree.tree_.n_leaves:
            raise RuntimeError(f'len(leave_indices) {len(leave_indices)} not equals to lc_dectree.tree_.n_leaves {lc_dectree.tree_.n_leaves} (some leaves have no sample!)')
        for leaf_idx in leave_indices:
            X_train_indices = np.where(X_leaf_indices==leaf_idx)[0]
            leave_values[leaf_idx] = np.mean(nn_prob[X_train_indices])
        return lc_dectree, leave_values

    ## get remained feature indices from masked feature indices 
    def _remain_feature_indices(self, feature_mask):
        all_indices = range(len(self.feature_names))
        remain_idx = [idx for idx in all_indices if idx not in feature_mask]
        # if self.debug: print('remain_idx',len(remain_idx))
        return remain_idx
    
    ## get decision from raw root regtree 
    # -1 means not into LC, otherwise means fall into LC (index in self.lc_leave_indices)
    def get_roottree_decision(self, 
            X_test, 
        ) -> np.ndarray:
        
        pred_leaf_idxs = self.roottree.apply(X_test)
        # _, leaf_indices = tree_utils.get_depth_and_leaves(self.roottree)
        
        lc_pos = -np.ones(len(self.roottree.tree_.n_node_samples)).astype(np.int)  # reverse map of self.lc_leave_indices (the POSITION of element), -1 means not lc leaf
        for i in range(len(self.lc_leave_indices)):
            lc_pos[self.lc_leave_indices[i]] = i
        
        def leaf2res(leaf_idx):
            return lc_pos[leaf_idx]

        roottree_res = np.asarray(list(map(leaf2res, pred_leaf_idxs)))
        return roottree_res

    ## get final predicetion of GEAD (normal/anomaly) ** BASED ON roottree_decision **
    def get_subtree_decision(self, 
            X_test, 
            roottree_res, # must get from self.get_roottree_decision()
            replace = False, # whether replace roottree_res
        ) -> np.ndarray:

        sub_indices = np.where(roottree_res!=-1)[0]
        if not replace: gead_pred = np.copy(roottree_res)
        else: gead_pred = roottree_res

        gead_pred[gead_pred==-1] = 0 # not into lc must be normal, but lc is not all abnormal
        
        batch_pred_map = dict() # key: idx (in self.lc_trees/...) of subtree, value: List[indices of samples fall into this subtree] 
        for idx in sub_indices:
            _lc_pos = gead_pred[idx] # get idx (in self.lc_trees/...) of subtree
            if _lc_pos not in batch_pred_map:
                batch_pred_map[_lc_pos] = [idx]
            else:
                batch_pred_map[_lc_pos].append(idx)

        if self.verbose: 
            print(f'Testing data size: {len(X_test)}, '
                  f'Fall into LC: {len(sub_indices)} ({int(len(sub_indices)/len(X_test)*100)}%), '
                  f'Total batch (subtrees): {len(batch_pred_map)} (cover: {int(len(batch_pred_map)/len(self.lc_leave_indices)*100)}%)') 

        if self.verbose: iter = tqdm(batch_pred_map, ncols=PRINT_LINEWIDTH)
        else: iter = batch_pred_map
        for _lc_pos in iter:
            lc_tree = self.lc_trees[_lc_pos] 
            feature_mask = self.lc_leave_data[_lc_pos]['decision_mask']
            feature_indices = np.asarray(self._remain_feature_indices(feature_mask))
            sample_indices = np.asarray(batch_pred_map[_lc_pos])
            pred_results = lc_tree.predict(X_test[sample_indices[:,np.newaxis],feature_indices])
            if self.subtree_type == 'reg':
                _pred_results = np.zeros_like(pred_results) # avoid replace error in the next two lines
                _pred_results[pred_results<self.value_thres] = 0
                _pred_results[pred_results>=self.value_thres] = 1
                pred_results = _pred_results
            gead_pred[sample_indices] = pred_results

        return gead_pred
    
    ## ONLY USED FOR ABLATION: get whether fall into NN_LC results (as a baseline), 0:into, 1:not
    def get_into_nnlc_decision(self,
            reg_results, # must get from self.get_regtree_decision()
            replace = False, # whether replace reg_results
        )-> np.ndarray:
        sub_indices = np.where(reg_results!=-1)[0]
        if not replace: into_nnlc = np.copy(reg_results)
        else: into_nnlc = reg_results

        INTO_FLAG = np.max(into_nnlc) + 1
        for idx in sub_indices:
            lc_num = self.lc_leave_indices[reg_results[idx]]
            if lc_num in self.nn_lc_leave_indices:
                into_nnlc[idx] = INTO_FLAG
            elif lc_num not in self.dt_lc_leave_indices:
                raise RuntimeError(f'FATAL ERROR: {lc_num} is not in DT_LC or NN_LC')
        
        into_nnlc[into_nnlc!=INTO_FLAG] = 0
        into_nnlc[into_nnlc==INTO_FLAG] = 1
        return into_nnlc

    ## ONLY USED FOR ABLATION: get whether fall into DT_LC results (as a baseline), 0:into, 1:not
    def get_into_dtlc_decision(self,
            raw_results, # must get from self.get_regtree_decision()
            replace = False, # whether replace reg_results
        )-> np.ndarray:
        sub_indices = np.where(raw_results!=-1)[0]
        if not replace: into_dtlc = np.copy(raw_results)
        else: into_dtlc = raw_results

        INTO_FLAG = np.max(into_dtlc) + 1
        for idx in sub_indices:
            lc_num = self.lc_leave_indices[raw_results[idx]]
            if lc_num in self.dt_lc_leave_indices:
                into_dtlc[idx] = INTO_FLAG
            elif lc_num not in self.nn_lc_leave_indices:
                raise RuntimeError(f'FATAL ERROR: {lc_num} is not in DT_LC or NN_LC')
        
        print('into_dtlc:', len(into_dtlc[into_dtlc==INTO_FLAG]))
        into_dtlc[into_dtlc!=INTO_FLAG] = 0
        into_dtlc[into_dtlc==INTO_FLAG] = 1
        return into_dtlc

    ## ONLY USED FOR ABLATION: get whether fall into LC results (whether NN_LC or DT_LC), 0:into, 1:not
    def get_into_lc_decision(self,
            reg_results, # must get from self.get_regtree_decision()
            replace = False, # whether replace reg_results
        )-> np.ndarray:

        if not replace: into_lc = np.copy(reg_results)
        else: into_lc = reg_results
        print('into_lc:', len(into_lc[into_lc!=-1]))
        into_lc[into_lc!=-1] = 1
        into_lc[into_lc==-1] = 0 
        return into_lc

    def _get_merged_tree_dict(self):
        root_tree, subtrees, subtree_indices = self.roottree, self.lc_trees, self.lc_leave_indices
        
        subtree_pos = dict()  # reverse map of subtree_indices (the POSITION in subtrees) e.g., subtree_indices=[3,34,56], subtree_pos={3:0, 34:1, 56:2}
        for i in range(len(subtree_indices)):
            subtree_pos[subtree_indices[i]] = i

        idx_inc = 0
        nodes = []
        merged_subtree_indices = [] # new subtree_indices in merged tree 
        # values = []

        def walk_tree(tree, node, level, idx, tree_idx, tree_feature): 
            # different from walk_tree() in get_format_tree(), this func will traverse root and subtrees
            # tree: root_tree or sub_tree
            # node: node_idx in root_tree or sub_tree
            # level: node level in merged tree
            # tree_idx: -1: root tree, >=0: idx in subtrees (self.lc_trees/self.lc_leave_indices)
            # tree_feature: feature idx in merged tree (if subtree, need remap)
            if self.debug: print(f'walk_tree(node={node}, level={level}, idx={idx}, tree_idx={tree_idx})')
            """Recursively iterates through all nodes in given decision tree and returns them as a list."""
            left = tree.tree_.children_left[node]
            right = tree.tree_.children_right[node]

            nonlocal idx_inc
            if left != right:  # if not  leaf node
                idx_inc += 1
                left = walk_tree(tree, left, level + 1, idx_inc, tree_idx, tree_feature)
                idx_inc += 1
                right = walk_tree(tree, right, level + 1, idx_inc, tree_idx, tree_feature)

            if tree_idx == -1 and (node in subtree_pos): # if subtree leaf
                subtree = subtrees[subtree_pos[node]]
                feature_mask = self.lc_leave_data[subtree_pos[node]]['decision_mask']
                feature_indices = np.asarray(self._remain_feature_indices(feature_mask))
                subtree_feature = np.copy(subtree.tree_.feature)   # get original subtree feature, copy in order to not replace
                remap_indices = np.where(subtree_feature!=TREE_UNDEFINED)[0] # exclude -2 (leaf node feature)
                subtree_feature[remap_indices] = feature_indices[subtree_feature[remap_indices]] # map subtree feature to whole tree / merged tree feature 
                merged_subtree_indices.append(idx_inc) # may not leaves in merged tree
                walk_tree(subtree, 0, level, idx_inc, subtree_pos[node], subtree_feature)
                
            else:
                node_data = {
                    "idx": idx,
                    "node": node,
                    "left": left,
                    "right": right,
                    "level": level,
                    "feature": tree_feature[node],
                    "threshold": tree.tree_.threshold[node],
                    "impurity": tree.tree_.impurity[node],
                    "samples": tree.tree_.n_node_samples[node],
                    "values": tree.tree_.value[node],
                    "weighted_samples": tree.tree_.weighted_n_node_samples[node],
                    "dtlc_value": None, # only valid for subtree nodes and when subtree is DT
                } 

                # node["values"] likes: array([[3.52546154]]) for RT-based subtrees, array([[99.,  6.]]) for DT-based subtrees
                # {is subtree} AND {subtree is DT} (will compute `dtlc_value`, which is a rt-like value)
                if tree_idx >= 0 and isinstance(tree, DecisionTreeClassifier):
                    _left_idx, _right_idx = tree.tree_.children_left[node], tree.tree_.children_right[node]
                    if _left_idx == _right_idx: # {is subtree leaf}
                        dtlc_value = self.dtlc_leave_values[tree_idx][node]
                    else: # non-leaf in subtree, get the average of two childs
                        _, dtlc_value = get_non_leaf_value(tree, node, self.dtlc_leave_values[tree_idx])
                    node_data["dtlc_value"] = np.asarray([[dtlc_value]])
                nodes.append(node_data)

            return idx
        
        # get non leaf rt-like value (will recursive get two child value and compute average) 
        def get_non_leaf_value(tree, node, dtlc_leave_values):
            left = tree.tree_.children_left[node]
            right = tree.tree_.children_right[node] 
            if left != right:  # if not  leaf node
                left_sample, leaf_value = get_non_leaf_value(tree, left, dtlc_leave_values)
                right_sample, right_value = get_non_leaf_value(tree, right, dtlc_leave_values)
                return left_sample+right_sample, (leaf_value*left_sample + right_value*right_sample)/(left_sample+right_sample)
            else:  # if leaf node
                return tree.tree_.n_node_samples[node], dtlc_leave_values[node]
            

        walk_tree(root_tree, 0, 0, idx_inc, -1, root_tree.tree_.feature)

        node_dtype = [
            ("left_child", "<i8"),
            ("right_child", "<i8"),
            ("feature", "<i8"),
            ("threshold", "<f8"),
            ("impurity", "<f8"),
            ("n_node_samples", "<i8"),
            ("weighted_n_node_samples", "<f8"),
        ]
        node_ndarray = np.array([], dtype=node_dtype)
        node_values = []
        max_depth = 0
        for node in sorted(nodes, key=lambda x: x["idx"]):
            if node["level"] > max_depth:
                max_depth = node["level"]

            node_ndarray = np.append(
                node_ndarray,
                np.array(
                    [
                        (
                            node["left"],
                            node["right"],
                            node["feature"],
                            node["threshold"],
                            node["impurity"],
                            node["samples"],
                            node["weighted_samples"],
                        )
                    ],
                    dtype=node_dtype,
                ),
            )

            if node["dtlc_value"] is not None: # this means this node is from a DT subtree
                # check whether need corrent the inconsist leaf decision between original DT and dtlc_value (rt-like value)
                if self.dtlc_correct and len(node["values"][0]) == 2: # otherwise is 1, this is because subtree only have one root node
                    max_index = np.argmax(node["values"][0])
                    # print('node["dtlc_value"][0]', node["dtlc_value"][0])
                    if max_index == 0 and node["dtlc_value"][0][0] >= self.value_thres:
                        node["dtlc_value"] = np.asarray([[self.value_thres - 0.00001]])
                    if max_index == 1 and node["dtlc_value"][0][0] < self.value_thres:
                        node["dtlc_value"] = np.asarray([[self.value_thres + 0.00001]])
                node_values.append(node["dtlc_value"])
            else:
                node_values.append(node["values"])

        # print(node_values)

        value_ndarray = np.array(node_values, dtype=np.float64)
        merged_dict = {
            "max_depth": max_depth,
            "node_count": len(node_ndarray),
            "nodes": node_ndarray,
            "values": value_ndarray,
        }

        return merged_dict, merged_subtree_indices

    def get_merged_tree(self, X_train):
        # use default DecisionTreeRegressor params to get a full tree
        # NOTE: to merge subtrees into root tree, we must initilize a bigger tree than root tree, directly use root tree will induce memory error
        self.merged_tree = tree.DecisionTreeRegressor().fit(X_train, self.nn_prob_train) 
        merged_tree_dict, merged_subtree_indices = self._get_merged_tree_dict()
        self.merged_lc_leave_indices = merged_subtree_indices
        self.merged_tree.tree_.__setstate__(merged_tree_dict)

    ## get final predicetion of GEAD (normal/anomaly, regtree + subtree) 
    def get_mergedtree_decision(self, 
            X_test,
            pruned_tree=False, #whether use self.pruned_merged_tree, must already use post-pruning function (e.g.top_k_post_prune)
        ) -> np.ndarray:

        if pruned_tree:
            if not hasattr(self, 'pruned_merged_tree'):
                raise ValueError(f"self.pruned_merged_tree has not been created!")
            pred_results = self.pruned_merged_tree.predict(X_test)
        else:
            pred_results = self.merged_tree.predict(X_test)
        gead_results = np.zeros_like(pred_results)
        gead_results[pred_results<self.value_thres] = 0
        gead_results[pred_results>=self.value_thres] = 1
        
        return gead_results

    ## use top-k post pruning (in Trustee) 
    ## NOTE: only work on self.merged_tree
    def top_k_post_prune(self, top_k=10):
        if not hasattr(self, 'merged_tree'):
            raise ValueError(f"self.merged_tree has not been created!")
        self.pruned_merged_tree = tree_utils.top_k_prune(self.merged_tree, top_k)


# include some interfaces/usages built upon GEADBase
class GEADSeqUsage(GEADSeqBase):   

    def get_n_leaves(self):
        if self.merged_tree is None:
            raise RuntimeError('merged_tree not created!')
        return self.merged_tree.tree_.n_leaves
        
    # evaluate merged gead 
    def eval_gead_merged(self, X_valid_in, X_valid_out, 
                        pruned_tree=False, 
                        verbose=True):
        if verbose: 
            print_title('Evaluation on Validation set (GEAD)')
        X_valid = seq2tab(X_valid_in, X_valid_out)
        nn_pred, _ = self.nn_ad.test_sequence_thres(X_valid_in, X_valid_out, ad_thres=self.ad_thres)

        gead_pred = self.get_mergedtree_decision(X_valid, pruned_tree)
        f_acc = eval_utils.fidelity_acc(gead_pred,nn_pred, verbose=verbose)
        p_con, n_con = eval_utils.consistency(gead_pred,nn_pred, verbose=verbose)
        p_cre, n_cre = eval_utils.credibility(gead_pred,nn_pred, verbose=verbose)
        return f_acc, p_con, n_con, p_cre, n_cre


    ## building GEAD in a most basic way
    def build(self, X_train_in, X_train_out):
        X_train_ben = seq2tab(X_train_in, X_train_out)

        print_title('1. Building Root Regression Tree and Identifying Low-confidence Leaves')
        self.get_roottree(X_train_ben)

        print_title('2. Data Augmentation for Low-confidence Leaves')
        self.lc_data_augment(X_train_ben)

        print_title('3. Building Low-confidence Fine-grained Subtrees')
        self.get_lc_trees(X_train_ben)

        print_title('4. Merging Subtrees into Roottree -> Merged Tree')
        self.get_merged_tree(X_train_ben)

        print_title('5. Binarizing Merged Tree')
        self.merged_tree = tree_utils.tree_binarization(self.merged_tree, self.value_thres)

    def build_params(self, X_train_in, X_train_out,
                     lc_params, aug_params, aug_settings,
                     subtree_params=None):
        
        self.set_params(**lc_params)
        self.set_params(**aug_params)

        X_train_ben = seq2tab(X_train_in, X_train_out)

        if self.verbose: print_title('1. Building Root Regression Tree and Identifying Low-confidence Leaves')
        self.get_roottree(X_train_ben)

        if self.verbose: print_title('2. Data Augmentation for Low-confidence Leaves')
        self.lc_data_augment(X_train_ben, **aug_settings)

        if self.verbose: print_title('3. Building Low-confidence Fine-grained Subtrees')
        if subtree_params is not None: self.get_lc_trees(X_train_ben, **subtree_params)
        else: self.get_lc_trees(X_train_ben)

        if self.verbose: print_title('4. Merging Subtrees into Roottree -> Merged Tree')
        self.get_merged_tree(X_train_ben)

        if self.verbose: print_title('5. Binarizing Merged Tree')
        self.merged_tree = tree_utils.tree_binarization(self.merged_tree, self.value_thres)        

    def build_params_scratch(self, X_train_in, X_train_out,
                     lc_params, aug_params, aug_settings,
                     subtree_params=None):
        
        self.set_params(**lc_params)
        self.set_params(**aug_params)

        X_train_ben = seq2tab(X_train_in, X_train_out)

        print_title('1. Building Root Regression Tree and Identifying Low-confidence Leaves')
        self.get_roottree(X_train_ben)

        print_title('2. Data Augmentation for Low-confidence Leaves')
        self.lc_data_augment(X_train_ben, **aug_settings)

        print_title('3. Building Fine-grained Trees from scratch')
        X_lc_concat = None
        for i, lcd in enumerate(self.lc_leave_data):
            if i==0: X_lc_concat = lcd['X_aug']
            else: X_lc_concat = np.concatenate((X_lc_concat,lcd['X_aug']))
        X_lc_concat = np.concatenate((X_train_ben, X_lc_concat))

        X_train_in, X_train_out = tab2seq(X_lc_concat, self.nn_ad.window_size)
        _, nn_prob_train = self.nn_ad.test_sequence_thres(X_train_in, X_train_out, ad_thres=self.ad_thres)
        nn_prob_train = nn_prob_train * self.prob_scale
        if subtree_params is not None: regressor = tree.DecisionTreeRegressor(**subtree_params)
        else: regressor = tree.DecisionTreeRegressor(**self.tree_params)
        self.merged_tree = regressor.fit(X_lc_concat, nn_prob_train)

        print_title('4. Binarizing Merged Tree')
        self.merged_tree = tree_utils.tree_binarization(self.merged_tree, self.value_thres)        
    
    ## building GEAD WITHOUT merging subtrees and roottree
    def build_unmerged(self, X_train_ben):
        
        if self.subtree_type == 'reg':
            raise NotImplementedError('subtree binarization is not implemented (will cause wrong rule number)')

        print_title('1. Building Root Regression Tree and Identifying Low-confidence Leaves')
        self.get_roottree(X_train_ben)

        print_title('2. Data Augmentation for Low-confidence Leaves')
        self.lc_data_augment(X_train_ben)

        print_title('3. Building Low-confidence Fine-grained Subtrees')
        self.get_lc_trees(X_train_ben)

        print_title('4. Merging Subtrees into Roottree -> Merged Tree')
        self.get_merged_tree(X_train_ben)

        print_title('5. Binarizing Root Tree')
        # NOTE:format_tree=False bacause otherwise the  self.lc_leave_indices will be changed
        self.roottree = tree_utils.tree_binarization(self.roottree, self.value_thres, self.lc_leave_indices, format_tree=False)
    
    ## Input a set of samples, Output the distribution of them falling into different leaves
    def apply_leaf_dist(self, X_test_in, X_test_out,
            highlight_text=None, # see tree_utils.TreePlotter highlight, if highlight_text is Not none, will reture `highlight` dict
            highlight_rate=0.2, # available only if highlight_text is not None, ignore leaf node idx whose samples rate < highlight_rate
        ):
        X_test = seq2tab(X_test_in, X_test_out)
        res = self.merged_tree.apply(X_test)
        res_dict = {}
        for r in res:
            if r in res_dict:
                res_dict[r] += 1 
            else:
                res_dict[r] = 1
        tree_ = self.merged_tree.tree_
        highlight = {}
        for nid in res_dict:
            sample_rate = res_dict[nid]/len(X_test)
            print(f'node id:{nid}, Samples:{res_dict[nid]}({sample_rate:.3f}), Value: {float(tree_.value[nid][0]):.3f}(thres:{self.ad_thres:.3f})')
            if highlight_text is not None and sample_rate >= highlight_rate:
                highlight[nid] = highlight_text
        if highlight_text is not None: 
            print('highlight', highlight)
        return highlight