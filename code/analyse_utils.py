'''
utils code for anlayzing based on GEAD
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import copy
from AE import epoches, lr, weight_decay, batch_size, percentage, device
from AE import autoencoder, criterion, getMSEvec, se2rmse, test, eval_roc, TPR_FPR
from general_utils import DEVICE

PLOT_CFGS = {
    "col_3":{ # 3 figs in 1 col
        'figsize':(6., 4.),
        'title': 35,
        'legend': 30,
        'ticks': 20,
    },
}

# plotting single feature distribution between normal_feat and anomal_feat
# NOTE: normal_feat and anomal_feat should be denormalized !
def plot_feat_distrib(
        normal_feat, anomal_feat,
        bin_num = 10,
        MIN = None, 
        MAX = None, 
        normal_legend = 'Normal', 
        anomal_legend = 'Abnormal', 
        color_1 = 'lightblue', 
        color_2 = 'lightcoral', 
        savefig = None, 
        title = None, 
        config = 'col_3',
    ): 

    CFG = PLOT_CFGS[config]

    # sns.set(style="white")
    plt.style.use('ggplot')
    
    if MIN is None:
        MIN = np.min(np.concatenate((normal_feat, anomal_feat)))
    if MAX is None:
        MAX = np.max(np.concatenate((normal_feat, anomal_feat)))
    
    bin_array = np.linspace(MIN, MAX, bin_num+1)
    cres = list(np.histogram(normal_feat, bins=bin_array))
    cres[0] = cres[0]/np.sum(cres[0])
    tres = list(np.histogram(anomal_feat, bins=bin_array))
    tres[0] = tres[0]/np.sum(tres[0])

    x = (cres[1][:-1] + cres[1][1:]) / 2
    width = x[1]-x[0]
    plt.figure(figsize=CFG['figsize'])
    
    plt.bar(x, tres[0], width=width, alpha=0.6, ec='black', label=anomal_legend, color=color_2)
    plt.bar(x, cres[0], width=width, alpha=0.6, ec='black', label=normal_legend, color=color_1)
    
    def get_smooth_axis(res):
        x = (res[1][:-1] + res[1][1:]) / 2
        x = np.insert(x,0,0.)
        x = np.insert(x,len(x),1.)
        y = res[0]
        y = np.insert(y,0,0.)
        y = np.insert(y,len(y),0.)
        print(x, y)
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 300)
        Y_ = X_Y_Spline(X_)
        return X_, Y_

    for bar in plt.gca().patches:
        bar.set_linewidth(0)
    plt.xlim(MIN-0.05, MAX+0.05)
    plt.xticks([])
    plt.legend(fontsize=CFG['legend'])
    plt.tick_params(axis='both', which='major', labelsize=CFG['ticks'])
    plt.gca().tick_params(axis='y', colors='white')
    plt.tight_layout() 
    plt.subplots_adjust(left=0.0, bottom=0.01, right=0.99, top=0.85)
    if title is not None:
        plt.title(title, fontsize=CFG['title'])
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


# get loss, auc, and model list for each epoch, used by M vs M - When to stop / Hyperparams
def each_epoch_loss_auc_model(X_train, feature_size, X_valid, y_valid, 
        epoches=epoches, lr=lr, weight_decay=weight_decay, verbose=True, thres_criter='rmse'):
    loss_list = []
    model_list = [] 
    auc_list = []

    model = autoencoder(feature_size, thres_criter).to(device)
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train = torch.from_numpy(X_train).type(torch.float).to(DEVICE)
    torch_dataset = Data.TensorDataset(X_train, X_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    for epoch in range(epoches):
        model.train()
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

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
        model.thres = o_thres

        if verbose:
            print('epoch:{}/{}'.format(epoch,epoches), '|Loss:', loss.item(), '|AUC:', roc_auc)
    

        loss_list.append(loss.item())
        model_list.append(copy.deepcopy(model))
        auc_list.append(roc_auc)

    return loss_list, auc_list, model_list