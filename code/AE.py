'''
autoencoder based anomaly detection model
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
# plt.switch_backend('Agg')

from general_utils import DEVICE

epoches = 20
lr = 1e-4
weight_decay = 1.e-7
batch_size = 1024
percentage = 0.99

device = DEVICE
criterion = nn.MSELoss()
getMSEvec = nn.MSELoss(reduction='none')


class autoencoder(nn.Module):
    def __init__(self, feature_size, criter='rmse'): # criter = ['rmse', 'mse'] 
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75), int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(feature_size*0.1),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75),int(feature_size)),
                                     )
        
        self.thres = np.Inf
        self.criter = criter
        print(f'NOTICE: use {self.criter} as the criteration of reconstruction error')

    def forward(self, x):
        encode = self.encoder(x)
        # print('encode', encode)
        decode = self.decoder(encode)
        # print('decode', decode)
        return decode
    
    # update anomaly detection threshold
    def update_thres(self, thres):
        self.thres = thres


def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])

def train(X_train, feature_size, epoches=epoches, lr=lr, percentage=percentage, weight_decay=weight_decay, verbose=True, thres_criter='rmse'):
    config = {
        'epoches': epoches,
        'lr':lr,
        'percentage':percentage,
        'weight_decay':weight_decay,
        'device':device,
    }
    if verbose:
        print("Hyper parameter config:", config)
        
    model = autoencoder(feature_size, thres_criter).to(device)
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    torch_dataset = Data.TensorDataset(X_train, X_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
            # if step % 10 == 0 :
        if verbose:
            print('epoch:{}/{}'.format(epoch,epoches), '|Loss:', loss.item())
    
    model.eval()
    output = model(X_train)
    if thres_criter == 'rmse':
        mse_vec = getMSEvec(output,X_train)
        rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
        if verbose:
            print("max AD score",max(rmse_vec))
        thres = max(rmse_vec)
        rmse_vec.sort()
        pctg = percentage
        thres = rmse_vec[int(len(rmse_vec)*pctg)]
    elif thres_criter == 'mse':
        mse_vec = torch.mean(getMSEvec(X_train, output), dim=1).cpu().data.numpy()
        if verbose:
            print("max AD score",max(mse_vec))
        thres = max(mse_vec)
        mse_vec.sort()
        pctg = percentage
        thres = mse_vec[int(len(mse_vec)*pctg)]
    else:
        print('Unknown criterion for selecting threshold', thres_criter)
        exit(-1)
        
    model.thres = thres
    if verbose:
        print(f"Threshold is: {thres} (at percentage {percentage})" )

    return model

# use X_valid to decide the best model
def train_valid(X_train, feature_size, X_valid, y_valid, 
                epoches=epoches, lr=lr, weight_decay=weight_decay, batch_size=batch_size, 
                percentage=0.99, thres_criter='rmse', verbose=True, debug=False, opt='Adam'):
    config = {
        'epoches': epoches,
        'lr':lr,
        'percentage':percentage,
        'weight_decay':weight_decay,
        'device':device,
    }
    if verbose:
        print("Hyper parameter config:", config)
        
    model = autoencoder(feature_size, thres_criter).to(device)
    if opt == 'Adam':
        optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizier = optim.SGD(model.parameters(), lr=lr)
    else: 
        raise RuntimeError(f'Unknown optimizier: {opt}')

    X_train = torch.from_numpy(X_train).type(torch.float).to(device)  
    torch_dataset = Data.TensorDataset(X_train, X_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    best_model, best_auc, best_thres =  None, -np.Inf, None

    for epoch in range(epoches):
        model.train()
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        if verbose:
            print('epoch:{}/{}'.format(epoch,epoches), '|Loss:', loss.item())

        model.eval()
        # get raw threshold for model 
        output = model(X_train)
        if thres_criter == 'rmse':
            mse_vec = getMSEvec(output,X_train)
            rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
            thres = max(rmse_vec)
            rmse_vec.sort()
            pctg = percentage
            thres = rmse_vec[int(len(rmse_vec)*pctg)]
        elif thres_criter == 'mse':
            mse_vec = torch.mean(getMSEvec(X_train, output), dim=1).cpu().data.numpy()
            thres = max(mse_vec)
            mse_vec.sort()
            pctg = percentage
            thres = mse_vec[int(len(mse_vec)*pctg)]
        else:
            print('Unknown criterion for selecting threshold', thres_criter)
            exit(-1)
        model.thres = thres

        _, y_valid_rmse = test(model, X_valid)
        roc_auc, o_thres = eval_roc(y_valid_rmse, y_valid, thres_max=model.thres*1.5, plot=False, verbose=False)
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_thres = o_thres
            best_model = copy.deepcopy(model)
            model.thres = best_thres
            y_valid_pred, _ = test(model, X_valid)
            tpr, fpr = TPR_FPR(y_valid_pred, y_valid, model.thres, verbose=False)
            if verbose: print(f'- update model! valid: roc auc is {best_auc:.5f}, tpr: {tpr:.5f}, fpr:{fpr:.5f}')
        elif debug:
            y_valid_pred, _ = test(model, X_valid)
            tpr, fpr = TPR_FPR(y_valid_pred, y_valid, o_thres, verbose=False)
            print(f'(- NOT update, valid: roc auc is {roc_auc:.5f}, tpr: {tpr:.5f}, fpr:{fpr:.5f})')

    if verbose:
        print(f"Threshold is: {model.thres} (at percentage {percentage})" )

    return best_model



def train_pos_sampling(X_train_ben, X_train_pos, feature_size, pos_weight = 1.,
                       epoches=epoches, lr=lr, weight_decay=weight_decay, batch_size=batch_size, 
                       percentage=0.99, thres_criter='rmse', verbose=True, debug=False, opt='Adam',
                       X_valid = None, y_valid = None, # if not None, will use validation set to choose best model
                       ):
    # clip to make sure X_train_pos and X_train_ben is with the same size
    if len(X_train_pos) >= len(X_train_ben):
        X_train_pos = X_train_pos[:len(X_train_ben)]
    else:
        repeats = -(-len(X_train_ben)//len(X_train_pos))
        X_train_pos = np.tile(X_train_pos, (repeats, 1))
        X_train_pos = X_train_pos[:len(X_train_ben)]

    config = {
        'epoches': epoches,
        'lr':lr,
        'percentage':percentage,
        'weight_decay':weight_decay,
        'device':device,
    }
    if verbose:
        print("Hyper parameter config:", config)
        
    model = autoencoder(feature_size, thres_criter).to(device)
    if opt == 'Adam':
        optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizier = optim.SGD(model.parameters(), lr=lr)
    else: 
        raise RuntimeError(f'Unknown optimizier: {opt}')

    X_train_ben = torch.from_numpy(X_train_ben).type(torch.float).to(device)
    X_train_pos = torch.from_numpy(X_train_pos).type(torch.float).to(device)

    torch_dataset = Data.TensorDataset(X_train_ben, X_train_pos)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    best_model, best_auc, best_thres =  None, -np.Inf, None
    Bound = nn.ReLU()
    MAX_THRES = 4.
    
    for epoch in range(epoches):
        model.train()
        for step, (batch_x_neg, batch_x_pos) in enumerate(loader):
            output_neg = model(batch_x_neg)
            loss_neg = criterion(output_neg, batch_x_neg)
            output_pos = model(batch_x_pos)
            loss_pos = Bound(MAX_THRES-criterion(output_pos, batch_x_pos))
            loss = loss_neg + pos_weight * loss_pos
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

        if verbose:
            print('epoch:{}/{}'.format(epoch,epoches), '|Loss:', loss.item(), f'(neg: {loss_neg.item()}, pos:{loss_pos.item()})')

        if (X_valid is not None and y_valid is not None) or (epoch == epoches-1):
            model.eval()
            # get raw threshold for model 
            output = model(X_train_ben)
            if thres_criter == 'rmse':
                mse_vec = getMSEvec(output,X_train_ben)
                rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
                thres = max(rmse_vec)
                rmse_vec.sort()
                pctg = percentage
                thres = rmse_vec[int(len(rmse_vec)*pctg)]
            elif thres_criter == 'mse':
                mse_vec = torch.mean(getMSEvec(X_train_ben, output), dim=1).cpu().data.numpy()
                thres = max(mse_vec)
                mse_vec.sort()
                pctg = percentage
                thres = mse_vec[int(len(mse_vec)*pctg)]
            else:
                print('Unknown criterion for selecting threshold', thres_criter)
                exit(-1)
            model.thres = thres

        if X_valid is not None and y_valid is not None:
            _, y_valid_rmse = test(model, X_valid)
            roc_auc, o_thres = eval_roc(y_valid_rmse, y_valid, thres_max=model.thres*1.5, plot=False, verbose=False)
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_thres = o_thres
                best_model = copy.deepcopy(model)
                model.thres = best_thres
                y_valid_pred, _ = test(model, X_valid)
                tpr, fpr = TPR_FPR(y_valid_pred, y_valid, model.thres, verbose=False)
                if verbose: print(f'- update model! valid: roc auc is {best_auc:.5f}, tpr: {tpr:.5f}, fpr:{fpr:.5f}')
            elif debug:
                y_valid_pred, _ = test(model, X_valid)
                tpr, fpr = TPR_FPR(y_valid_pred, y_valid, o_thres, verbose=False)
                print(f'(- NOT update, valid: roc auc is {roc_auc:.5f}, tpr: {tpr:.5f}, fpr:{fpr:.5f})')
        else:
            best_model = model

    if verbose:
        print(f"Threshold is: {model.thres} (at percentage {percentage})" )

    return best_model


@torch.no_grad()
def test(model, X_test, thres=None):
    if thres is None:
        thres = model.thres 
    model.eval()
    X_test = torch.from_numpy(X_test).type(torch.float)    
    X_test = X_test.to(device)
    output = model(X_test)
    if model.criter == 'rmse':
        mse_vec = getMSEvec(output,X_test)
        rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
        idx_mal = np.where(rmse_vec>thres)
        ano_score = rmse_vec
    elif model.criter == 'mse':
        mse_vec = torch.mean(getMSEvec(X_test, output), dim=1).cpu().data.numpy()
        idx_mal = np.where(mse_vec>thres)
        ano_score = mse_vec
    else:
        raise NotImplementedError
    
     
    y_pred = np.asarray([0] * len(ano_score))
    y_pred[idx_mal] = 1

    return y_pred, ano_score


def test_plot(rmse_vec, thres, file_name = None, label = None):
    plt.figure()
    plt.plot(np.linspace(0,len(rmse_vec)-1,len(rmse_vec)),[thres]*len(rmse_vec),c='black',label='99th-threshold')
    # plt.ylim(0,thres*2.)

    if label is not None:
        idx = np.where(label==0)[0]
        plt.scatter(idx, rmse_vec[idx], s=8, color='blue', alpha=0.4, label='Normal')
        
        idx = np.where(label==1)[0]
        plt.scatter(idx, rmse_vec[idx], s=8, color='red', alpha=0.7, label='Anomalies')
    else:
        plt.scatter(np.linspace(0,len(rmse_vec)-1,len(rmse_vec)),rmse_vec,s=8,alpha=0.4, label='Test samples' )
    
    plt.legend()
    plt.xlabel('Sample NO.')
    plt.ylabel('Anomaly Score (RMSE)')
    plt.title('Per-sample Score')
    if file_name is None:
        plt.show()
    else:
        plt.rcParams.update({'figure.dpi':300})
        plt.savefig(file_name)



def TPR_FPR(y_prob, y_true, thres, verbose=True): 
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.where(y_prob >= thres, 1, 0)

    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = (fp / (fp + tn + 1e-10))
    tpr = (tp / (tp + fn + 1e-10))
    
    if verbose:
        print('TPR:', tpr, 'FPR:', fpr,)
        print('TN:', tn, 'TP:', tp, 'FP:', fp, 'FN:', fn)
        
    return tpr, fpr


def multi_fpr_tpr(y_prob, y_true, thres_max, thres_min=0, split = 300, is_P_mal=True): 
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fpr = []
    tpr = []

    thresholds = np.linspace(thres_min, thres_max, split)
    for threshold in thresholds:
        if is_P_mal: 
            y_pred = np.where(y_prob >= threshold, 1, 0)
        else:
            y_pred = np.where(y_prob <= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        # print('fp+tn', fp+tn, 'tp + fn', tp + fn)
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr, tpr, thresholds

def eval_roc(probs, labels, thres_max, thres_min=0, split=300, is_P_mal=True, plot=True, verbose=True):
    fprs, tprs, thresholds = multi_fpr_tpr(probs, labels, thres_max, thres_min=thres_min, split=split, is_P_mal=is_P_mal)
    roc_auc = metrics.auc(fprs, tprs)
    if verbose: print('roc_auc:',roc_auc)
    
    if plot:
        plt.figure()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fprs, tprs, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    
    optimal_idx = np.argmax(np.asarray(tprs) - np.asarray(fprs))
    optimal_threshold = thresholds[optimal_idx]
    return roc_auc, optimal_threshold

class Normalizer:
    def __init__(self, train_data, clip=False, delta=1e-10):
        self.train_min = np.min(train_data, axis=0)
        self.train_max = np.max(train_data, axis=0)
        self.clip = clip
        self.delta = delta

    def transform(self, data):
        return (data - self.train_min) / (self.train_max - self.train_min + self.delta)

    def denorm(self, data):
        return data * (self.train_max - self.train_min) + self.train_min

    def denorm_query(self, index, norm_value):
        range = self.train_max[index] - self.train_min[index]
        return norm_value * range + self.train_min[index]