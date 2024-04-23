'''
utils of handling decision/regression trees
'''

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import sys
from sklearn.tree import DecisionTreeRegressor
import copy
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

import AE


## get node depth list and leaves index list for given regtree
def get_depth_and_leaves(regtree):
    n_nodes, children_left, children_right = regtree.tree_.node_count, regtree.tree_.children_left, regtree.tree_.children_right
    node_depth = np.zeros(n_nodes, dtype=np.int64)
    is_leaves = np.zeros(n_nodes, dtype=bool)
    stack = [(0, -1)]  # node index and father index 
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if children_left[node_id] != -1:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    leaves_indices = np.where(is_leaves)[0]
    return node_depth, leaves_indices


## get AD threshold from nn_pctg (for deciding LC of DNN)
def get_value_thres(model, X_train_norm, nn_pctg):
    _, y_train_rmse = AE.test(model, X_train_norm)
    y_train_rmse.sort()
    value_thres = y_train_rmse[int(len(y_train_rmse)*nn_pctg)]
    return value_thres


## get impurity threshold from dt_pctg (for deciding LC of Tree)
def get_impurity_thres(regtree, dt_pctg):
    impurity = regtree.tree_.impurity.reshape(-1)
    n_node_samples = regtree.tree_.n_node_samples
    _, leaves_indices = get_depth_and_leaves(regtree)
    impurity = impurity[leaves_indices]
    n_node_samples = n_node_samples[leaves_indices]
    argidx = np.argsort(impurity)
    impurity = impurity[argidx]
    n_node_samples = n_node_samples[argidx]
    cnt = 0
    # print(impurity)
    # print(n_node_samples)
    for i in range(len(n_node_samples)):
        cnt += n_node_samples[i]
        if cnt > np.sum(n_node_samples) * dt_pctg:
            return impurity[i]

## given a regtree, get all leaf indices and the unique feature used from root to the leaf 
def get_leaf_info(regtree):
    leaf_indices, unique_feature_numer = [], []
    
    def find_leaf_paths(tree, node_id=0, current_path=None, unique_features=None):
        nonlocal leaf_indices, unique_feature_numer

        if current_path is None:
            current_path = []
        if unique_features is None:
            unique_features = set()

        current_path.append(node_id)

        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Reached a leaf node
            features_in_path = set(tree.feature[i] for i in current_path if tree.feature[i] >= 0)
            unique_features.update(features_in_path)
            leaf_indices.append(node_id)
            unique_feature_numer.append(len(features_in_path))

        else:
            # Recursively traverse left and right subtrees
            find_leaf_paths(tree, tree.children_left[node_id], current_path, unique_features)
            find_leaf_paths(tree, tree.children_right[node_id], current_path, unique_features)

        current_path.pop()
    
    find_leaf_paths(regtree.tree_)
    return leaf_indices, unique_feature_numer


def get_LC_leaves(regtree, value_thres, impurity_thres, 
                  verbose = True,
                ):
    impurity = regtree.tree_.impurity.reshape(-1)
    # n_node_samples = regtree.tree_.n_node_samples
    value = regtree.tree_.value.reshape(-1)
    DT_LC_leaves_indices = []
    NN_LC_leaves_indices = []

    # leaves_indices = np.where(regtree.tree_.children_left == -1)[0]
    leaves_indices, unique_feature_numer = get_leaf_info(regtree)
    for i, lidx in enumerate(leaves_indices):
        if unique_feature_numer[i] >= regtree.tree_.n_features: # can not be LC if all features are already used (appear mostly in seq mode)
            continue
        if impurity[lidx] >= impurity_thres:
            DT_LC_leaves_indices.append(lidx)
        if value[lidx] >= value_thres:
            NN_LC_leaves_indices.append(lidx)
    LC_leaves_indices = list(set(DT_LC_leaves_indices+NN_LC_leaves_indices))
    if verbose:
        print(f'(LC Leaves Number) Total: {len(LC_leaves_indices)}, DT(impurity):{len(DT_LC_leaves_indices)}, NN(value): {len(NN_LC_leaves_indices)}')
    return LC_leaves_indices, DT_LC_leaves_indices, NN_LC_leaves_indices


# Iterates through the given Decision Tree to collect updated tree node structure
# From https://github.com/TrusteeML/trustee/blob/a0a17bdd9038608711f8a0bfcc36d6812aaa311f/trustee/utils/tree.py#L26
def get_dt_dict(tree):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    samples = tree.tree_.n_node_samples
    weighted_samples = tree.tree_.weighted_n_node_samples
    impurity = tree.tree_.impurity
    values = tree.tree_.value

    idx_inc = 0
    nodes = []
    # values = []

    def walk_tree(node, level, idx):
        """Recursively iterates through all nodes in given decision tree and returns them as a list."""
        # print(f'walk_tree(node={node}, level={level}, idx={idx})')
        left = children_left[node]
        right = children_right[node]

        nonlocal idx_inc
        if left != right:  # if not  leaf node
            idx_inc += 1
            left = walk_tree(left, level + 1, idx_inc)
            idx_inc += 1
            right = walk_tree(right, level + 1, idx_inc)

        nodes.append(
            {
                "idx": idx,
                "node": node,
                "left": left,
                "right": right,
                "level": level,
                "feature": features[node],
                "threshold": thresholds[node],
                "impurity": impurity[node],
                "samples": samples[node],
                "values": values[node],
                "weighted_samples": weighted_samples[node],
            }
        )

        return idx

    walk_tree(0, 0, idx_inc)

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
        node_values.append(node["values"])
    value_ndarray = np.array(node_values, dtype=np.float64)

    dt_dict = {
        "max_depth": max_depth,
        "node_count": len(node_ndarray),
        "nodes": node_ndarray,
        "values": value_ndarray,
    }

    return dt_dict
    
## generate format tree according the struture of sklearn.tree 
## General Workflow: operation(sklearn.tree) -> unformat_tree, reset_tree(unformat_tree) -> format tree 
##  - an example of `operation` function is tree_binarization()
def get_format_tree(
        unformat_tree, # raw self-designed tree
        replace = True, # output tree whether replace input tree 
    ):
    
    if replace:
        format_tree = unformat_tree
    else:
        format_tree = copy.deepcopy(unformat_tree)
    # orgin_dict = copy.deepcopy(unformat_tree.tree_.__getstate__()) 
    format_tree.tree_.__setstate__(get_dt_dict(unformat_tree))
    # format_tree.tree_.__setstate__(orgin_dict)
    return format_tree


## merge two child nodes if their decisions are both normal or abnormal 
def tree_binarization(
        original_tree:DecisionTreeRegressor, # original regtree to be binarized
        threshold:float, # threshold for anomaly detection
        skipped_leaves:List[int]=[], # common use case is binarization of root tree, skipped_leaves = self.lc_leave_indices
        format_tree = True, # whether call get_format_tree() at the end to generate format tree
        verbose = True,
    ):
    n_merge = 0 # count how many times two leaves are merged into one

    # copy regtree and get meta-info arrays
    new_tree = copy.deepcopy(original_tree)
    tree_ = new_tree.tree_
    # n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    threshold_values = tree_.threshold
    values = tree_.value
    feature = tree_.feature

    # judge whether a node is a leaf
    def is_leaf(node_id, tree_):
        if tree_.children_left[node_id] == tree_.children_right[node_id] == TREE_LEAF:
            return True
        else:
            return False
    
    # recursive pruning function
    def dfs_prune(node_id):
        nonlocal n_merge

        # return if leaf
        if is_leaf(node_id,tree_): return 

        # post-order (root-last) traversal
        dfs_prune(children_left[node_id])
        dfs_prune(children_right[node_id])
        
        if children_left[node_id] == children_right[node_id]:
            raise RuntimeError('Unexpected Fatal Error! Only non-leaf node can enter this process!')
        
        # merge condition: (123 are AND relationship)
        # 1. two child value are both >= OR < threshold; 
        # 2. two child are both leaves; 
        # 3. two child are both NOT LC leaves
        if ((values[children_left[node_id]][0][0] >= threshold and \
                values[children_right[node_id]][0][0] >= threshold) or \
            (values[children_left[node_id]][0][0] < threshold and \
                values[children_right[node_id]][0][0] < threshold)) and \
            (is_leaf(children_left[node_id],tree_) and is_leaf(children_right[node_id],tree_)) and \
            (children_left[node_id] not in skipped_leaves and children_right[node_id] not in skipped_leaves):

            n_merge += 1
            # set the merged node as leaf and update realted meta-info arrays
            children_left[node_id] = TREE_LEAF
            children_right[node_id] = TREE_LEAF
            values[node_id][0] = original_tree.tree_.value[node_id][0] #np.mean([values[children_left[node_id]][0], values[children_right[node_id]][0]], axis=0)
            feature[node_id] = original_tree.tree_.feature[node_id]
            threshold_values[node_id] = original_tree.tree_.threshold[node_id]

    # begin pruning
    dfs_prune(0)
    
    # NOTE: cannot use tree_.n_leaves property to count leaf number of new tree 
    # NOTE: merged nodes are still -1 but not really deteled!
    if verbose:
        print(f'NOTICE: Finish Tree Binarization: totally merge {n_merge} times')

    if format_tree:
        new_tree = get_format_tree(new_tree)
        if original_tree.tree_.n_leaves - new_tree.tree_.n_leaves != n_merge:
            raise RuntimeError('{orginal_tree.tree_.n_leaves} (original_tree) - {new_tree.tree_.n_leaves} (new_tree) != {n_merge} (n_merge)')
    
    return new_tree


## simulate pre-prune with min_impurity by post pruning (used by Trustee, note Trustee names this as 'max'_impurity, but called 'min' is more suitable)
## NOTE: we say "**simulate** pre-prune" because min_impurity pre-prune is not offically supported by sklearn (support `min_impurity_decrease` but not min_impurity !)
def min_impurity_prune(
        original_tree,
        min_impurity=0.1, # if impurity <= min_impurity, will stop split (0.1 is the default value given by Trustee)
        format_tree = True, # strongly recommanded to be True
    ):

    new_tree = copy.deepcopy(original_tree)
    tree_ = new_tree.tree_
    impurity = tree_.impurity
    children_left = tree_.children_left
    children_right = tree_.children_right   

    def stop_splitting(node):
        _impurity = impurity[node]

        if _impurity <= min_impurity: # stop splitting 
            children_left[node] = TREE_LEAF
            children_right[node] = TREE_LEAF
        else:
            left_child = children_left[node]
            if left_child != TREE_LEAF:
                stop_splitting(left_child)
            
            right_child = children_right[node]
            if right_child != TREE_LEAF:
                stop_splitting(right_child)

    stop_splitting(0)

    if format_tree:
        new_tree = get_format_tree(new_tree)
        print(f'(min_impurity_prune) totally pruning {original_tree.tree_.n_leaves - new_tree.tree_.n_leaves} leafs')
    else:
        raise RuntimeWarning('format_tree is set as False, which is not recommanded and may induce expected error!')
    
    return new_tree

## NOTE: Copy from Trustee
def prune_index(dt, index, prune_level): # NOTE: make the pruned node's as the leaf node (instead of del this node)
    """Prunes the given decision tree at the given index and returns the number of pruned nodes"""
    if index < 0:
        return 0

    left_idx = dt.tree_.children_left[index]
    right_idx = dt.tree_.children_right[index]

    # turn node into a leaf by "unlinking" its children
    dt.tree_.children_left[index] = TREE_LEAF if prune_level == 0 else TREE_UNDEFINED
    dt.tree_.children_right[index] = TREE_LEAF if prune_level == 0 else TREE_UNDEFINED

    # if there are shildren, visit them as well
    if left_idx != TREE_LEAF and right_idx != TREE_LEAF:
        prune_index(dt, left_idx, prune_level + 1)
        prune_index(dt, right_idx, prune_level + 1)


## NOTE: Copy from Trustee
def get_dt_info(dt):
    """Iterates through the given Decision Tree to collect relevant information."""
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature
    thresholds = dt.tree_.threshold
    values = dt.tree_.value
    samples = dt.tree_.n_node_samples
    impurity = dt.tree_.impurity

    splits = []
    features_used = {}

    def walk_tree(node, level, path):
        """Recursively iterates through all nodes in given decision tree and returns them as a list."""
        if children_left[node] == children_right[node]:  # if leaf node
            node_class = np.argmax(values[node][0]) if len(np.array(values[node][0])) > 1 else values[node][0][0]
            node_prob = (
                (values[node][0][node_class] / np.sum(values[node][0])) * 100
                if np.array(values[node][0]).ndim > 1
                else 0
            )
            return [
                {
                    "level": level,
                    "path": path,
                    "class": node_class,
                    "prob": node_prob,
                    "samples": samples[node],
                }
            ]

        feature = features[node]
        threshold = thresholds[node]
        left = children_left[node]
        right = children_right[node]

        if feature not in features_used:
            features_used[feature] = {"count": 0, "samples": 0}

        features_used[feature]["count"] += 1
        features_used[feature]["samples"] += samples[node]

        splits.append(
            {
                "idx": node,
                "level": level,
                "feature": feature,
                "threshold": threshold,
                "samples": samples[node],
                "values": values[node],
                "gini_split": (impurity[left], impurity[right]),
                "data_split": (np.sum(values[left]), np.sum(values[right])),
                "data_split_by_class": [
                    (c_left, c_right) for (c_left, c_right) in zip(values[left][0], values[right][0])
                ],
            }
        )

        return walk_tree(left, level + 1, path + [(node, feature, "<=", threshold)]) + walk_tree(
            right, level + 1, path + [(node, feature, ">", threshold)]
        )

    branches = walk_tree(0, 0, [])
    return features_used, splits, branches

## NOTE: reference from Trustee, Prunes a given decision tree down to its top-k branches, sorted by number of samples covered
def top_k_prune(dt, top_k):
    _, nodes, branches = get_dt_info(dt)
    top_branches = sorted(branches, key=lambda p: p["samples"], reverse=True)[:top_k]
    # for i, tb in enumerate(top_branches):
    #     print(f'{i:>3}', tb,'\n')

    prunned_dt = copy.deepcopy(dt)

    nodes_to_keep = set({})
    for branch in top_branches:
        for (node, _, _, _) in branch["path"]:
            nodes_to_keep.add(node)

    for node in nodes:
        if node["idx"] not in nodes_to_keep:
            prune_index(prunned_dt, node["idx"], 0)

    # update classifier with prunned model
    prunned_dt.tree_.__setstate__(get_dt_dict(prunned_dt))

    return prunned_dt


## get max and average depth (among all rules) of a given tree
def depth_statistics(tree, verbose=True, inter=True):
    children_left = tree.children_left
    children_right = tree.children_right
    
    def dfs(node, depth):
        if children_left[node] == children_right[node]: # leaf 
            return [depth]
        
        # else dfs left and right children 
        left_depths = dfs(children_left[node], depth + 1) if children_left[node] != children_right[node] else []
        right_depths = dfs(children_right[node], depth + 1) if children_left[node] != children_right[node] else []
        
        return left_depths + right_depths
    
    # begin recursive
    depths = dfs(0, 1)

    mean_depth, max_depth = np.mean(depths), max(depths)

    if inter:
        mean_depth= int(mean_depth)
    if verbose:
       print(f'Rule length: MEAN:{mean_depth}, MAX:{max_depth}')
    
    return mean_depth, max_depth


## customize tree ploting
class TreePlotter:
    def __init__(self, tree, feature_names, max_depth=None):
        self.tree = tree
        self.feature_names = feature_names
        self.max_depth=max_depth

        self._figsize_estimator()

    def _figsize_estimator(self):
        n_leaves = self.tree.tree_.n_leaves
        if n_leaves < 20:
            figsize=(20,10)
        elif n_leaves < 100:
            figsize=(35,20)
        else:
            figsize=(50,35)
        self.figsize = figsize
    
    # show binarized tree for REGRESSION tree by matplotlib 
    # NOTE: deprecated by plot_bin_tree()
    def plot_bin_tree_matplotlib(self, ad_thres, node_ids=True, save_pth=None):
        _, ax = plt.subplots(figsize=self.figsize)
        tree.plot_tree(self.tree, feature_names=self.feature_names, node_ids=node_ids, proportion=True, impurity=False)
        labels = [t.get_text() for t in ax.texts]

        if node_ids: n_leaf_lines = 3 # (node id, samples/proportion, pred_value) # NOTE: leaf does not have feature name
        else: n_leaf_lines = 2
        # replace reg tree results with binary results
        for i, label in enumerate(labels):
            lines = label.split("\n")
            if len(lines) == n_leaf_lines:
                value_line = lines[-1]
                value = float(value_line.split("value =")[1])
                if value > ad_thres:
                    new_label = label.replace(f"value = {value}", "Anomaly")
                else:
                    new_label = label.replace(f"value = {value}", "Normal")
                labels[i] = new_label
        
        ax.clear()
        tree.plot_tree(self.tree, filled=True, ax=ax, max_depth=self.max_depth, proportion=True, rounded=True,  node_ids=node_ids, fontsize=15)
        for i, text in enumerate(ax.texts):
            text.set_text(labels[i])
        
        plt.show()
        if save_pth is not None:
            plt.savefig(save_pth)

    # show **binarized** reg tree for REGRESSION tree by graphviz(dot)
    def plot_reg_tree(self, ad_thres, save_pth=None, 
            branch_condition=False, # whether show condition at branch instead of at node (<= value)
            denormalized=True, # whether denormalize the conditional threshold (not ad_thres) to show original value
            normalizer=None, # must exist if denormalized=True
            highlight={}, # Dict {leaf idx <int>:attack label <str>, ...} for each leaf idx in this dict, will highlight the path from root to it
        ):

        if denormalized and normalizer is None:
            raise RuntimeError('normalizer must be set if denormalized=True!')

        tree_ = self.tree.tree_
        feature_name = [
            self.feature_names[i] if i != tree._tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        
        max_value = np.max(tree_.value)

        # global config 
        dot_file = "digraph Tree {\n"
        dot_file += 'ranksep=0.4;\n'
        dot_file += 'node [shape=ellipse, style="filled, rounded", fontname="Arial", color=transparent];\n'
        dot_file += 'edge [fontname="Arial", color=lightgrey, dir=both, arrowhead=dot, arrowtail=dot, arrowsize=0.2];\n' #  decorate=true

        # save all highlight nodes and edges
        path_nodes = set()
        path_edges = set()

        # find path to leaf node
        def find_path(node, leaf_node_index, path=[]):
            if node == leaf_node_index:
                path_nodes.update(path + [node])
                return True
            if node == -1 or node in path:
                return False

            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]

            if find_path(left_child, leaf_node_index, path + [node]):
                path_edges.add((node, left_child))
                return True
            if find_path(right_child, leaf_node_index, path + [node]):
                path_edges.add((node, right_child))
                return True
            return False
        
        if len(highlight) > 0:
            for hidx in highlight:
                if not find_path(0, hidx):
                    raise RuntimeError(f'Cannot find leave with index {hidx}')

        def recurse(node, depth):
            nonlocal dot_file
            indent = "  " * depth
            if tree_.children_left[node] != tree_.children_right[node]:
                name = feature_name[node]
                threshold_value = tree_.threshold[node]
                if denormalized:
                    _normalized_feat = np.zeros(len(self.feature_names)) # other dimensions do not matter for denormalization
                    feature_index = tree_.feature[node]
                    # print('feature_index', feature_index)
                    _normalized_feat[feature_index] = threshold_value
                    threshold_value = normalizer.denorm(_normalized_feat)[feature_index]
                value = tree_.value[node].max()
                gray_scale = int(255 - 127 * (value / max_value))
                gray_hex = f'#{gray_scale:02x}{gray_scale:02x}{gray_scale:02x}'

                # non-leaf node
                if node in path_nodes:
                    _label = f"<<B>{name}</B>>"
                    _color='red'
                    _penwidth = 2
                else:
                    _label = '"'+name+'"'
                    _color='transparent'
                    _penwidth = 1
                if branch_condition:
                    dot_file += f'{indent}{node} [label={_label}, fillcolor="{gray_hex}", color="{_color}", penwidth={_penwidth}];\n'
                else:
                    dot_file += f'{indent}{node} [label="{_label} &le; {threshold_value:.2f}", fillcolor="{gray_hex}, color="{_color}, penwidth={_penwidth}"];\n'
                left_child = tree_.children_left[node]
                right_child = tree_.children_right[node]

                # left branch
                left_samples = tree_.n_node_samples[left_child]
                right_samples = tree_.n_node_samples[right_child]
                if (node, left_child) in path_edges:
                    _color = 'LightCoral'
                    _label = f"<<B>&le; {threshold_value:.2f}</B>>"
                    _fontcolor = "black"
                else:
                    _color = 'lightgrey'
                    _label = f'"&le; {threshold_value:.2f}"'
                    _fontcolor = '#808080'
                penwidth = 1.0 + 5.0 * left_samples / (left_samples + right_samples)
                if branch_condition: 
                    dot_file += f'{indent}{node} -> {left_child} [tailport="sw", headport="n", penwidth={penwidth}, color ={_color}, xlabel={_label}, fontcolor="{_fontcolor}"];\n'
                else:
                    dot_file += f'{indent}{node} -> {left_child} [tailport="sw", headport="n", penwidth={penwidth}, color ={_color}, fontcolor={_fontcolor}];\n'
                recurse(left_child, depth + 1)

                # right branch
                if (node, right_child) in path_edges:
                    _color = 'LightCoral'
                    _label = f"<<B>&gt; {threshold_value:.2f}</B>>"
                    _fontcolor = "black"
                else:
                    _color = 'lightgrey'
                    _label = f'"&gt; {threshold_value:.2f}"'
                    _fontcolor = '#808080'
                penwidth = 1.0 + 5.0 * right_samples / (left_samples + right_samples)
                if branch_condition:
                    dot_file += f'{indent}{node} -> {right_child} [tailport="se", headport="n", penwidth={penwidth}, color ={_color}, xlabel={_label}, fontcolor="{_fontcolor}"];\n'
                else:
                    dot_file += f'{indent}{node} -> {right_child} [tailport="se", headport="n", penwidth={penwidth}, color ={_color}, fontcolor={_fontcolor}];\n'
                recurse(right_child, depth + 1)

            else:  # leaf node
                node_value = tree_.value[node][0][0]
                _label = "Anomaly" if node_value > ad_thres else "Normal"
                _fillcolor = "LightCoral" if _label == "Anomaly" else "LightSkyBlue"
                _color = 'transparent'
                _fontcolor = 'black'
                _penwidth = 1
                if node in highlight: 
                    _fontcolor = '#A52A2A' if _label == "Anomaly" else "#00008B" # cover old _fontcolor if highlight 
                    if highlight[node] is not "": 
                        _label = f'<<B>{highlight[node]}</B>>' # cover old _label if highlight
                    else: _label = f'<<B>{_label}</B>>'
                    _color = 'red' # cover old _color if highlight
                    _penwidth = 2 # cover old _penwidth if highlight
                
                dot_file += f'{indent}{node} [label={_label}, fillcolor="{_fillcolor}", color="{_color}", fontcolor="{_fontcolor}", shape=box, style="filled, rounded", penwidth={_penwidth}];\n'

        recurse(0, 1)
        dot_file += "}"

        graph = graphviz.Source(dot_file)
        if save_pth is not None:
            graph.render(save_pth, view=True)

        '''
        In jupyter, use the following code to show graph:

        from IPython.display import display
        display(graph)
        '''
        return graph

    # show dec tree for REGRESSION tree by graphviz(dot), mainly used for baseline
    def plot_dec_tree(self, save_pth=None, 
            branch_condition=False, # whether show condition at branch instead of at node (<= value)
            denormalized=True, # whether denormalize the conditional threshold (not ad_thres) to show original value
            normalizer=None, # must exist if denormalized=True
            highlight={}, # Dict {leaf idx <int>:attack label <str>, ...} for each leaf idx in this dict, will highlight the path from root to it
        ):

        if denormalized and normalizer is None:
            raise RuntimeError('normalizer must be set if denormalized=True!')

        tree_ = self.tree.tree_
        feature_name = [
            self.feature_names[i] if i != tree._tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        # global config 
        dot_file = "digraph Tree {\n"
        dot_file += 'ranksep=0.4;\n'
        dot_file += 'node [shape=ellipse, style="filled, rounded", fontname="Arial", color=transparent];\n'
        dot_file += 'edge [fontname="Arial", color=lightgrey, dir=both, arrowhead=dot, arrowtail=dot, arrowsize=0.2];\n' #  decorate=true

        # save all highlight nodes and edges
        path_nodes = set()
        path_edges = set()

        # find path to leaf node
        def find_path(node, leaf_node_index, path=[]):
            if node == leaf_node_index:
                path_nodes.update(path + [node])
                return True
            if node == -1 or node in path:
                return False

            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]

            if find_path(left_child, leaf_node_index, path + [node]):
                path_edges.add((node, left_child))
                return True
            if find_path(right_child, leaf_node_index, path + [node]):
                path_edges.add((node, right_child))
                return True
            return False
        
        if len(highlight) > 0:
            for hidx in highlight:
                if not find_path(0, hidx):
                    raise RuntimeError(f'Cannot find leave with index {hidx}')

        def recurse(node, depth):
            nonlocal dot_file
            indent = "  " * depth
            if tree_.children_left[node] != tree_.children_right[node]:
                name = feature_name[node]
                threshold_value = tree_.threshold[node]
                if denormalized:
                    _normalized_feat = np.zeros(len(self.feature_names)) # other dimensions do not matter for denormalization
                    feature_index = tree_.feature[node]
                    _normalized_feat[feature_index] = threshold_value
                    threshold_value = normalizer.denorm(_normalized_feat)[feature_index]

                class_counts = tree_.value[node][0]
                proportion = class_counts[0] / sum(class_counts) 
                gray_scale = max(int(255 * proportion - 32), 127)
                gray_hex = f'#{gray_scale:02x}{gray_scale:02x}{gray_scale:02x}'

                # non-leaf node
                if node in path_nodes:
                    _label = f"<<B>{name}</B>>"
                    _color='red'
                    _penwidth = 2
                else:
                    _label = '"'+name+'"'
                    _color='transparent'
                    _penwidth = 1
                if branch_condition:
                    dot_file += f'{indent}{node} [label={_label}, fillcolor="{gray_hex}", color="{_color}", penwidth={_penwidth}];\n'
                else:
                    dot_file += f'{indent}{node} [label="{_label} &le; {threshold_value:.2f}", fillcolor="{gray_hex}, color="{_color}, penwidth={_penwidth}"];\n'
                left_child = tree_.children_left[node]
                right_child = tree_.children_right[node]
                
                # left branch
                left_samples = tree_.n_node_samples[left_child]
                right_samples = tree_.n_node_samples[right_child]
                if (node, left_child) in path_edges:
                    _color = 'LightCoral'
                    _label = f"<<B>&le; {threshold_value:.2f}</B>>"
                    _fontcolor = "black"
                else:
                    _color = 'lightgrey'
                    _label = f'"&le; {threshold_value:.2f}"'
                    _fontcolor = "#808080"
                penwidth = 1.0 + 5.0 * left_samples / (left_samples + right_samples)
                if branch_condition: 
                    dot_file += f'{indent}{node} -> {left_child} [tailport="sw", headport="n", penwidth={penwidth}, color ={_color}, xlabel={_label}, fontcolor="{_fontcolor}"];\n'
                else:
                    dot_file += f'{indent}{node} -> {left_child} [tailport="sw", headport="n", penwidth={penwidth}, color ={_color}, fontcolor="{_fontcolor}"];\n'
                recurse(left_child, depth + 1)

                # right branch
                if (node, right_child) in path_edges:
                    _color = 'LightCoral'
                    _label = f"<<B>&gt; {threshold_value:.2f}</B>>"
                    _fontcolor = "black"
                else:
                    _color = 'lightgrey'
                    _label = f'"&gt; {threshold_value:.2f}"'
                    _fontcolor = "#808080"
                penwidth = 1.0 + 5.0 * right_samples / (left_samples + right_samples)
                if branch_condition:
                    dot_file += f'{indent}{node} -> {right_child} [tailport="se", headport="n", penwidth={penwidth}, color ={_color}, xlabel={_label}, fontcolor="{_fontcolor}"];\n'
                else:
                    dot_file += f'{indent}{node} -> {right_child} [tailport="se", headport="n", penwidth={penwidth}, color ={_color}, fontcolor="{_fontcolor}"];\n'
                recurse(right_child, depth + 1)

            else:  # leaf node
                class_counts = tree_.value[node][0]
                _label = "Normal" if class_counts[0] >= class_counts[1] else "Anomaly"
                _fillcolor = "LightCoral" if _label == "Anomaly" else "LightSkyBlue"
                _color = 'transparent'
                _fontcolor = 'black'
                _penwidth = 1
                if node in highlight: 
                    _fontcolor = '#A52A2A' if _label == "Anomaly" else "#00008B" # cover old _fontcolor if highlight 
                    if highlight[node] is not "": _label = f'<<B>{highlight[node]}</B>>' # cover old _label if highlight
                    else: _label = f'<<B>{_label}</B>>'
                    _color = 'red' # cover old _color if highlight
                    _penwidth = 2 # cover old _penwidth if highlight
                
                dot_file += f'{indent}{node} [label={_label}, fillcolor="{_fillcolor}", color="{_color}", fontcolor="{_fontcolor}", shape=box, style="filled, rounded", penwidth={_penwidth}];\n'

        recurse(0, 1)
        dot_file += "}"
        
        graph = graphviz.Source(dot_file)
        if save_pth is not None:
            graph.render(save_pth, view=True)

        '''
        In jupyter, use the following code to show graph:

        from IPython.display import display
        display(graph)
        '''
        return graph
