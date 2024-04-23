'''
utils for data preprocess
'''
from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import random 
import torch
import os
import sys
import _pickle as pkl
from collections import defaultdict

from AE import Normalizer

## save model in AE
def save_model(model, pth):
    write_security_check(pth)
    pkl.dump(model, open(pth, 'wb'))
    print(f'Successfully save model at {pth}!')

## load model in AE 
def load_model(pth):
    model = pkl.load(open(pth, 'rb'))
    return model

## check whether overwirte before saving files
def write_security_check(filename, fail_exit=True):
    if os.path.isfile(filename):
        confirm = input(f"Security Check: {filename} already exists, overlapping? (Input 'y' to Confirm) ")
        if confirm.lower() != "y":
            print("**WARNING**: Canceled Writing Operation!")
            if fail_exit: sys.exit(-1)
        else:
            print("**WARNING**: Confirm Overlapping!")
            return


## fix random seed of numpy, random, and torch package
def set_random_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

## process nan -> 0 and +- inf -> max/min (of current column) for pd.df
def replace_na_inf_values(df:pd.DataFrame) -> pd.DataFrame:
    
    new_df = df.fillna(0)
    max_vals = {col: new_df[col].replace([np.inf, -np.inf], np.nan).max(skipna=True) for col in new_df.columns}
    min_vals = {col: new_df[col].replace([np.inf, -np.inf], np.nan).min(skipna=True) for col in new_df.columns}
    for i, _ in new_df.iteritems():
        new_df[i].replace(np.inf, max_vals[i],inplace=True)
        new_df[i].replace(-np.inf, min_vals[i],inplace=True)
    
    return new_df


class DataLoaderBase:
    def __init__(self, verbose=True):
        self.feature_name = None
        self.verbose = verbose

    # load X, y (np.ndarray) from file(s)
    # NOTE: must be override by subclasses
    def load_data(self, 
        ):
        raise NotImplementedError
    
    ## get all X, y from csv, without normalize
    ## support sampling to avoid too much data
    # NOTE: return dataset is NOT normalized ! split is NOT shuffled!
    def data_split(self, X_all, y_all, 
                   split_rate=1. # rate of the first (usually used as 'valid set')
                   ):

        assert 0.<=split_rate<=1.
        X, y  = shuffle(X_all, y_all, random_state = 0)
        X_list, y_list = [], []
        X_list.append(X[:int(len(X_all)*split_rate)])
        y_list.append(y[:int(len(y_all)*split_rate)])
        X_list.append(X[int(len(X_all)*split_rate):])
        y_list.append(y[int(len(y_all)*split_rate):])

        if self.verbose: print(f"FIHISIHED: total {len(X)} items (ben: {len(y[y==0])}, mal: {len(y[y==1])})")
        
        if split_rate == 1.:
            return X_list[0], y_list[0]
        elif split_rate == 0.:
            return X_list[1], y_list[1]
        else:
            return X_list[0], y_list[0], X_list[1], y_list[1]

    # get split dataset, normalizer, and feature name
    # NOTE: return dataset is ALREADY normalized !
    def data_split_norm(self, 
            X_all, y_all,
            n_train_ben:Union[int,float]=0.6,
            n_train:Union[int,float]=0.6,
            n_vali:Union[int,float]=0.2,
            n_test:Union[int,float]=-1, # if -1 , will automatically compute as 1-n_vali-n_train
            norm:str='train_ben',  # use ['train_ben', 'train', 'none'] to normalize (none: return un-norm data)
        ):
        dataset = {}

        if isinstance(n_train_ben, float): n_train_ben = int(n_train_ben*len(X_all))
        if isinstance(n_train, float): n_train = int(n_train*len(X_all))
        if isinstance(n_vali, float): n_vali = int(n_vali*len(X_all))
        if n_test == -1:
            n_test = len(X_all) - n_train - n_vali
        elif isinstance(n_test, float): n_test = int(n_test*len(X_all))

        assert n_train+n_vali+n_test <= len(X_all) # <= because the int() opt below will cause loss of float
        assert norm in ['train_ben', 'train', 'none']
        
        X_train_ben, y_train_ben = X_all[y_all==0][:n_train_ben], y_all[y_all==0][:n_train_ben]
        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_vali, y_vali = X_all[n_train:n_train+n_vali], y_all[n_train:n_train+n_vali]
        X_test, y_test = X_all[n_train+n_vali:], y_all[n_train+n_vali:]
        
        if (n_train+n_vali)>len(X_all):
            print(f'Warning: not enough samples for validation (expected:{len(n_vali)}, remain:{len(X_vali)})')
        if (n_train+n_vali+n_test)>len(X_all):
            print(f'Warning: not enough samples for test (expected:{len(n_test)}, remain:{len(X_test)})')

        class nonenorm: # for code brevity 
            def __init__(self) -> None:
                pass
            def transform(self, X):
                return X
            
        if norm == 'train_ben':
            normalizer = Normalizer(X_train_ben)
        elif norm == 'train':
            normalizer = Normalizer(X_train)
        elif norm == 'none':
            normalizer = nonenorm()
        
        X_train_ben = normalizer.transform(X_train_ben)
        X_train = normalizer.transform(X_train)
        X_vali = normalizer.transform(X_vali)
        X_test = normalizer.transform(X_test)
        X_all = normalizer.transform(X_all)
        
        if self.verbose:
            print(f'X_all:       {len(X_all)} (ben: {len(y_all[y_all==0])}, mal: {len(y_all[y_all==1])})')
            print(f'X_train_ben: {len(X_train_ben)} (ben: {len(y_train_ben[y_train_ben==0])}, mal: {len(y_train_ben[y_train_ben==1])})')
            print(f'X_train:     {len(X_train)} (ben: {len(y_train[y_train==0])}, mal: {len(y_train[y_train==1])})')
            print(f'X_vali:      {len(X_vali)} (ben: {len(y_vali[y_vali==0])}, mal: {len(y_vali[y_vali==1])})')
            print(f'X_test:      {len(X_test)} (ben: {len(y_test[y_test==0])}, mal: {len(y_test[y_test==1])})')

        dataset['X_all'], dataset['y_all'] = X_all, y_all.astype(int)
        dataset['X_train_ben'], dataset['y_train_ben'] = X_train_ben, y_train_ben.astype(int)
        dataset['X_train'], dataset['y_train'] = X_train, y_train.astype(int)
        dataset['X_vali'], dataset['y_vali'] = X_vali, y_vali.astype(int)
        dataset['X_test'], dataset['y_test'] = X_test, y_test.astype(int)
        return dataset, normalizer

class CicDataLoader(DataLoaderBase):

    def __init__(self, verbose=True, 
                 improved=False, # whether parsing improved dataset or not
            ):
        super(CicDataLoader, self).__init__(verbose)
        # training params
        self.improved = improved

    # override
    def load_data(self, 
            csv_pth:Union[str,List[str]], 
            shuffled=True, # whether shuffle the dataset, False will split the dataset in sample id order
            random_state=0, # use for shuffle
        ):
        if isinstance(csv_pth, str): 
            csv_pth = [csv_pth]
        if self.improved:
            X_all, y_all = np.empty((0, 78)), np.empty(0)
            for pth in csv_pth:
                df = pd.read_csv(pth) # 91 columns
                if self.feature_name is None:
                    self.feature_name = df.columns.tolist()[8:-5]
                df = replace_na_inf_values(df)
                y = df.values[:,-2]
                y[y!='BENIGN'] = 1
                y[y=='BENIGN'] = 0
                X = df.values[:,8:-5].astype(np.float64) # 78
                X_all = np.concatenate((X_all, X))
                y_all = np.concatenate((y_all, y))
                print(f"Processed {pth.split('/')[-1]}, total {len(X)} items (ben: {len(y[y==0])}, mal: {len(y[y==1])})")
        else:
            X_all, y_all = np.empty((0, 77)), np.empty(0)
            for pth in csv_pth:
                df = pd.read_csv(pth) # 79 columns
                if self.feature_name is None:
                    self.feature_name = df.columns.tolist()[1:-1]
                df = replace_na_inf_values(df)
                y = df.values[:,-1]
                y[y!='BENIGN'] = 1
                y[y=='BENIGN'] = 0
                X = df.values[:,1:-1].astype(np.float64)
                X_all = np.concatenate((X_all, X))
                y_all = np.concatenate((y_all, y))
                if self.verbose: print(f"Processed {pth.split('/')[-1]}, total {len(X)} items (ben: {len(y[y==0])}, mal: {len(y[y==1])})")
        
        if shuffled:
            X_all, y_all = shuffle(X_all, y_all, random_state = random_state)
        return X_all, y_all
    
    # given a label, load all X_label (e.g., get all DDoS label data)
    # NOTE: X_label is NOT normalized!
    def load_labeled_data(self, 
                csv_pth:Union[str,List[str]],
                label:Union[str,List[str]],
                shuffled=False,
                random_state=0,
            ):
        if isinstance(csv_pth, str): 
            csv_pth = [csv_pth]
        if isinstance(label, str):
            label = [label]
    
        if self.improved:
            X_label, y_label = np.empty((0, 78)), np.empty(0)
            for pth in csv_pth:
                df = pd.read_csv(pth) # 91 columns
                if self.feature_name is None:
                    self.feature_name = df.columns.tolist()[8:-5]
                df = replace_na_inf_values(df)
                labeled_rows = df[df.iloc[:, -2].isin(label)]
                X = labeled_rows.values[:,8:-5].astype(np.float64) # 78
                y = labeled_rows.values[:,-2]
                X_label = np.concatenate((X_label, X))
                y_label = np.concatenate((y_label, y))
                if self.verbose: print(f"Processed {pth.split('/')[-1]}, total {len(X)} items ({label} has {len(X)} items)")
        else:
            X_label, y_label = np.empty((0, 77)), np.empty(0)
            for pth in csv_pth:
                df = pd.read_csv(pth) # 79 columns
                if self.feature_name is None:
                    self.feature_name = df.columns.tolist()[1:-1]
                df = replace_na_inf_values(df)
                labeled_rows = df[df.iloc[:, -1].isin(label)]
                X = labeled_rows.values[:,1:-1].astype(np.float64) # 77
                y = labeled_rows.values[:,-1]
                X_label = np.concatenate((X_label, X))
                y_label = np.concatenate((y_label, y))
                if self.verbose: print(f"Processed {pth.split('/')[-1]}, total {len(X)} items ({label} has {len(X)} items)")

        if len(label) == 1:
            if shuffled:
                np.random.seed(random_state)
                np.random.shuffle(X_label)
            return X_label 
        else:
            if shuffled:
                X_label, y_label = shuffle(X_label, y_label, random_state = random_state)
            return X_label, y_label


class KitsuneDataLoader(DataLoaderBase):
    def __init__(self, feat_name_pth, # feature name (.npy) 
                 verbose=True,
            ):
        super(KitsuneDataLoader, self).__init__(verbose)
        self.feature_name = np.load(feat_name_pth)

    def load_data(self, 
            feat_pth:Union[str,List[str]], # feature file .npy path list 
            label_pth:Union[str,List[str]], # 0/1 label file .npy path list, must consistent w/ csv_pth
            shuffled=True, # whether shuffle the dataset, False will split the dataset in sample id order
            random_state=0, # use for shuffle
        ):
        if isinstance(feat_pth, str): 
            feat_pth = [feat_pth]
        if isinstance(label_pth, str):
            label_pth = [label_pth]
        assert len(feat_pth) == len(label_pth)
        
        X_all, y_all = np.empty((0, 100)), np.empty(0)
        for f, l in zip(feat_pth,label_pth):
            _X = np.load(f)
            _y = np.load(l)
            assert len(_X) == len(_y)
            X_all = np.concatenate((X_all, _X))
            y_all = np.concatenate((y_all, _y))
            print(f"Processed {f.split('/')[-1]}, total {len(_X)} items (ben: {len(_y[_y==0])}, mal: {len(_y[_y==1])})")
        
        if shuffled:
            X_all, y_all = shuffle(X_all, y_all, random_state = random_state)
        return X_all, y_all
    
    # different from cicids17, labels must be specified and assigned in this func
    def load_labeled_data(
            feat_pth:Union[str,List[str]], # feature file .npy path list 
            label_pth:Union[str,List[str]], # 0/1 label file .npy path list, must consistent w/ csv_pth
            assigned_label:Union[str,List[str]], # list of attack name str
            shuffled=False,
            random_state=0,
        ):

        if isinstance(feat_pth, str): 
            feat_pth = [feat_pth]
        if isinstance(label_pth, str):
            label_pth = [label_pth]
        if isinstance(assigned_label, str):
            assigned_label = [assigned_label]
        assert len(feat_pth) == len(label_pth) == len(assigned_label)
        
        X_label, y_label = np.empty((0, 100)), np.empty(0)
        for f, l, n in zip(feat_pth, label_pth, assigned_label):
            _X = np.load(f)
            _y = np.load(l)
            assert len(_X) == len(_y)
            _y[_y==0] = 'Normal'
            _y[_y==1] = n
            X_label = np.concatenate((X_label, _X))
            y_label = np.concatenate((y_label, _y))
            print(f"Processed {f.split('/')[-1]}, total {len(_X)} items (ben: {len(_y[_y==0])}, mal: {len(_y[_y==1])})")
        
        if shuffled:
            X_label, y_label = shuffle(X_label, y_label, random_state = random_state)
        return X_label, y_label


class KyotoDataLoader(DataLoaderBase):

    def __init__(self, dataset_folder=None,
                 verbose=True,
            ):
        super(KyotoDataLoader, self).__init__(verbose)
        if dataset_folder is None: 
            self.dataset_folder = '/data1/hdq/Projects/GEAD/dataset/Kyoto_Dec_100w/'
        else:
            self.dataset_folder = dataset_folder

    def load_data(self, 
            csv_name:Union[str,List[str]], 
            shuffled=True, # whether shuffle the dataset, False will split the dataset in sample id order
            random_state=0, # use for shuffle
        ):
        if isinstance(csv_name, str): 
            csv_name = [csv_name]
        
        X_all, y_all = np.empty((0, 48)), np.empty(0)
        for n in csv_name:
            pth = os.path.join(self.dataset_folder, f'{n}.csv')
            df = pd.read_csv(pth) # 49 columns
            if self.feature_name is None:
                self.feature_name = df.columns.tolist()[:-1]
            df = replace_na_inf_values(df)
            y = df.values[:,-1]
            y[y!='NORMAL'] = 1
            y[y=='NORMAL'] = 0
            X = df.values[:,:-1].astype(np.float64)
            X_all = np.concatenate((X_all, X))
            y_all = np.concatenate((y_all, y))
            if self.verbose: print(f"Processed {pth.split('/')[-1]}, total {len(X)} items (ben: {len(y[y==0])}, mal: {len(y[y==1])})")

        if shuffled:
                X_all, y_all = shuffle(X_all, y_all, random_state = random_state)
        return X_all, y_all
    

# NOTE: for HDFS dataset, data do NOT need normalized
class HDFSDataLoader(DataLoaderBase):

    def __init__(self, dataset_folder=None, window_size = None,
                 verbose=True,
            ):
        super(HDFSDataLoader, self).__init__(verbose)
        if dataset_folder is None: 
            self.dataset_folder = '/data1/hdq/Projects/GEAD/dataset/hdfs_5/'
            self.window_size = 5
        else:
            self.dataset_folder = dataset_folder
            self.window_size = window_size

    def load_data(self, 
            type='train', # ['train', 'test']
            shuffled=False, # whether shuffle the dataset, False will split the dataset in sample id order
            random_state=0, # use for shuffle
        ):

        assert type in ['train', 'test']
        if type == 'train':
            data = np.load(self.dataset_folder+'train_normal.npz')
            X_in, X_out, y = data['input'], data['output'], np.asarray([0]*len(data['output']))    
            assert np.sum(y) == 0.
            # session = None
        else:
            data = np.load(self.dataset_folder+'test.npz')
            X_in, X_out, y = data['input'], data['output'], data['label']
        
            if shuffled:
                random.seed(random_state) 
                indexes = list(range(len(X_in)))
                random.shuffle(indexes)
                X_in = [X_in[i] for i in indexes]
                X_out = [X_out[i] for i in indexes]
                y = [y[i] for i in indexes]
        
        if self.verbose:
            print(f'Load {len(X_in)} sequence ({type}) with window = {self.window_size} (Normal: {len(y[y==0])}, Abnormal:{len(y[y==1])})')
        
        return X_in, X_out, y # , session
    
    def data_split(self, X_in, X_out, y,  # session,
            split_rate=0.5 # rate of the first (usually used as 'valid set')
            ):
        assert 0.<split_rate<1.
        X_all = np.concatenate((X_in, np.reshape(X_out, (-1,1))), axis=1)
        X_1, y_1, X_2, y_2 = super(HDFSDataLoader, self).data_split(X_all, y, split_rate=split_rate) 
        X_in_1, X_out_1 =  X_1[:, :self.window_size], X_1[:, self.window_size]
        X_in_2, X_out_2 =  X_2[:, :self.window_size], X_2[:, self.window_size]

        if self.verbose:
            print(f'First  Part: Total {len(y_1)} samples (Normal:{len(y_1[y_1==0])}, Abnormal:{len(y_1[y_1==1])})' )
            print(f'Second Part: Total {len(y_2)} samples (Normal:{len(y_2[y_2==0])}, Abnormal:{len(y_2[y_2==1])})' )

        return (X_in_1, X_out_1, y_1), (X_in_2, X_out_2, y_2)