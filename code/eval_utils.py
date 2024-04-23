'''
utils for evaluation
'''
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import AE

## computing TPR, FPR, TP, FP, TN, FN
def eval_tpr_fpr(y_pred:np.ndarray, y_true:np.ndarray, verbose=True):
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fpr = (fp / (fp + tn + 1e-10))
    tpr = (tp / (tp + fn + 1e-10))
    if verbose:
        print('TPR:', tpr, 'FPR:', fpr)
        print('TN:', tn, 'TP:', tp, 'FP:', fp, 'FN:', fn)
    else:
        return tpr, fpr, fp, tp, fn, tn


## computing accuracy-like fidelity metric = (# same preds) / (# all samples)
def fidelity_acc(tree_pred:np.ndarray, nn_pred:np.ndarray, verbose = True):
    assert len(tree_pred) == len(nn_pred)
    same_preds = np.sum(tree_pred == nn_pred)
    fidelity = same_preds/len(tree_pred)
    if verbose: print(f'Overall Fidelity (ACC): [{fidelity:.4f}]')
    return fidelity

## Consistency: from the perspective of DNN, how many same pos/neg prediction with DNN can be made by the Tree?
def consistency(tree_pred:np.ndarray, nn_pred:np.ndarray, verbose=True):
    assert len(tree_pred) == len(nn_pred)
    tpr, fpr, fp, tp, fn, tn = eval_tpr_fpr(tree_pred, nn_pred, False)
    pos_consistency, neg_consistnecy = tpr, 1-fpr
    if verbose:
        print(f'(fp:{fp}, tp:{tp}, fn:{fn}, tn:{tn})')
        print(f'Positive Consistency: [{pos_consistency:.4f}], Negative Consistency: [{neg_consistnecy:.4f}]')
    return pos_consistency, neg_consistnecy

## Credibility: from the perspective of Tree, if the tree thinks a node is a pos/neg, to what extent (how much credibility) we can believe it?
def credibility(tree_pred:np.ndarray, nn_pred:np.ndarray, verbose = True):
    _, _, fp, tp, fn, tn = eval_tpr_fpr(tree_pred, nn_pred, False)
    pos_credibility = tp/(tp+fp)
    neg_credibility = tn/(tn+fn)
    if verbose:
        print(f'Positive Credibility: [{pos_credibility:.4f}], Negative Credibility: [{neg_credibility:.4f}]')
    return pos_credibility, neg_credibility


## evaluate fidelity by replacing high-depth nodes and see model performance change
def fidelity_replace(model, 
        X_test, y_test, #samples to be replaced 
        feat_dim=0, # how many (first) dims in feat_idx need to considered
        tree=None, # used for searching high-depth nodes
        feat_idx=[], # feature dim to be replaced, if default [], will research high-depth nodes
        replace='mean', # replace_method
        metric='auc', # which metric used to evaluate
        verbose=False,
    ):
    
    if len(feat_idx) > 0: 
        if feat_dim > 0: 
            feat_idx = feat_idx[:feat_dim]
    else:
        for f in tree.tree_.feature:
            if f != -2 and f not in feat_idx:
                feat_idx.append(f)
            if len(feat_idx) == feat_dim:
                break
    # print(feat_idx)
        
    nn_pred, nn_prob = AE.test(model, X_test)
    roc_auc, best_thres = AE.eval_roc(nn_prob, y_test, thres_max=model.thres*1.5, plot=verbose, verbose=verbose)
    
    X_new = np.copy(X_test)
    pos_feat = X_test[y_test==1][:, feat_idx]
    neg_feat = X_test[y_test==0][:, feat_idx]
    if replace=='mean':
        pos_mean = np.mean(pos_feat, axis=0)
        neg_mean = np.mean(neg_feat, axis=0)
        rows = y_test == 1
        X_new[np.ix_(rows, feat_idx)] = neg_mean
        rows = y_test == 0
        X_new[np.ix_(rows, feat_idx)] = pos_mean

    elif replace=='random':
        
        random_idx = np.random.randint(0, len(neg_feat), size=len(pos_feat))
        rows = y_test == 1
        X_new[np.ix_(rows, feat_idx)] = neg_feat[random_idx]
        
        random_idx = np.random.randint(0, len(pos_feat), size=len(neg_feat))
        rows = y_test == 0
        X_new[np.ix_(rows, feat_idx)] = pos_feat[random_idx]
        
    else:
        raise NotImplementedError
    
    nn_pred, nn_prob = AE.test(model, X_new)
    new_roc_auc, best_thres = AE.eval_roc(nn_prob, y_test, thres_max=model.thres*1.5, plot=verbose, verbose=verbose)
    
    if metric == 'auc':
        print(f"(Fidelity Evaluation) *{metric}* changes from {roc_auc:.3f} to {new_roc_auc:.3f} (\Delta:{roc_auc-new_roc_auc:.3f})")
    else:
        raise NotImplementedError
    