"""
This implementation of DeepLog is based on the open-source code at 
https://github.com/wuyifan18/DeepLog 
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from general_utils import DEVICE

class DeepLogAD:
    
    def __init__(self, 
                 window_size=5, 
                 num_layers=2, 
                 hidden_size=64, 
                 num_epochs=100, 
                 batch_size=2048, 
                 num_candidates=9,
                 lr = 1e-3,
                 num_classes=28, # Fixed for this hdfs dataset 
                 verbose=True,
                 device=None
                 ):
        if device is None: device = DEVICE

        self.window_size = window_size
        self.lr = lr
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_candidates = num_candidates
        self.device = device
        self.num_classes = num_classes
        self.verbose = verbose
        
        self.create_lstm()

    def create_lstm(self):
        self.model = LSTM_onehot(self.hidden_size, self.num_layers, self.num_classes).to(self.device)

    def train(self, input_seq, output_label):
        seq_dataset = TensorDataset(torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_label))
        dataloader = DataLoader(seq_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)

        # Train the model
        start_time = time.time()
        total_step = len(dataloader)
        for epoch in range(self.num_epochs):  # Loop over the dataset multiple times
            train_loss = 0
            for step, (seq, label) in enumerate(dataloader):
                seq = seq.clone().detach().view(-1, self.window_size).to(self.device)
                seq = F.one_hot(seq,num_classes=self.num_classes).float()
                output = self.model(seq, self.device)
                
                loss = criterion(output, label.to(self.device))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if self.verbose:
                print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, self.num_epochs, train_loss / total_step))
            
        elapsed_time = time.time() - start_time
        if self.verbose:
            print('elapsed_time: {:.3f}s'.format(elapsed_time))

    
    # sequence-wise prediction  (predict by ** whether label in predict logits candidate **)
    @torch.no_grad()
    def test_sequence(self, X_input, X_output, 
             test_batch = 5000000, # depends on the available GPU memory (~5GB for <test_batch>:20000, <num_classes>:1600, <window_size>:10)
             softmax = True,
             postive = True, # if true, we transform the normal prob score to abnormal score! (by 1-neg)
             ):
        self.model.eval()
        y_pred = []
        y_prob = []

        with torch.no_grad():
            ## batch fashion considering the limit of GPU memory
            test_steps = len(X_input)//test_batch
            if len(X_input) % test_batch != 0:
                test_steps += 1
            for i in range(test_steps):
                seq = torch.tensor(X_input[test_batch*i:test_batch*(i+1)], dtype=torch.long).view(-1, self.window_size).to(self.device)
                seq = F.one_hot(seq,num_classes=self.num_classes).float()
                label = torch.tensor(X_output[test_batch*i:test_batch*(i+1)]).view(-1).to(self.device)
                output = self.model(seq, self.device)
                predicted = torch.argsort(output, 1)[:, self.num_candidates:]
                for j in range(len(label)):
                    if label[j] in predicted[j]:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
                if softmax:
                    output = F.softmax(output, dim=1)
                probs = output.cpu().detach().numpy()[np.arange(len(X_output[test_batch*i:test_batch*(i+1)])),X_output[test_batch*i:test_batch*(i+1)]]
                if postive:
                    probs = 1. - probs
                if i == 0:
                    y_prob = probs
                else:
                    y_prob = np.concatenate((y_prob,probs))
                if self.verbose:
                    print('deeplog model testing: %d/%d'%(i,test_steps))
            
        return y_pred, y_prob
    
    # sequence-wise prediction (predict by ** a certain ad threshold **)
    @torch.no_grad()
    def test_sequence_thres(self, X_input, X_output, 
             test_batch = 5000000, # depends on the available GPU memory (~5GB for <test_batch>:20000, <num_classes>:1600, <window_size>:10)
             softmax = True, # if true, output logits will softmax (normalized to 0 and 1, summed as 1)
             postive = True, # if true, we transform the normal prob score to abnormal score! (by 1-neg)
             ad_thres = 0.9, # if postive = true && label logits >= ad_thres ==> positive (vice versa)
             ):
        self.model.eval()
        y_pred = []
        y_prob = []

        with torch.no_grad():
            ## batch fashion considering the limit of GPU memory
            test_steps = len(X_input)//test_batch
            if len(X_input) % test_batch != 0:
                test_steps += 1
            for i in range(test_steps):
                seq = torch.tensor(X_input[test_batch*i:test_batch*(i+1)], dtype=torch.long).view(-1, self.window_size).to(self.device)
                try:
                    seq = F.one_hot(seq,num_classes=self.num_classes).float()
                except Exception as e:
                    print("An error occurred:", str(e))
                    print('seq',seq)
                    exit(-1)
                # label = torch.tensor(X_output[test_batch*i:test_batch*(i+1)]).view(-1).to(self.device)
                output = self.model(seq, self.device)
                # predicted = torch.argsort(output, 1)[:, self.num_candidates:]
                if softmax:
                    output = F.softmax(output, dim=1)
                probs = output.cpu().detach().numpy()[np.arange(len(X_output[test_batch*i:test_batch*(i+1)])),X_output[test_batch*i:test_batch*(i+1)]]
                if postive:
                    probs = 1. - probs
                    preds = np.zeros_like(probs)
                    preds[probs<ad_thres] = 0
                    preds[probs>=ad_thres] = 1
                else:
                    preds = np.zeros_like(probs)
                    preds[probs<ad_thres] = 1
                    preds[probs>=ad_thres] = 0
                if i == 0:
                    y_prob = probs
                    y_pred = preds
                else:
                    y_prob = np.concatenate((y_prob,probs))
                    y_pred = np.concatenate((y_pred,preds))
                if self.verbose:
                    print('deeplog model testing: %d/%d'%(i,test_steps))
            
        return y_pred, y_prob
    
    # session-wise prediction (this is more suitable for model performance evaluation)
    @torch.no_grad()
    def eval_session(self, test_normal_loader, test_abnormal_loader):
        self.model.eval()
        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            for line in test_normal_loader:
                for i in range(len(line) - self.window_size):
                    seq = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq = torch.tensor(seq, dtype=torch.long).view(-1, self.window_size).to(self.device)
                    seq = F.one_hot(seq,num_classes=self.num_classes).float()
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = self.model(seq, self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        FP += 1
                        break

        with torch.no_grad():
            for line in test_abnormal_loader:
                for i in range(len(line) - self.window_size):
                    seq = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    if label == -1:
                        TP += 1
                        break
                    seq = torch.tensor(seq, dtype=torch.long).view(-1, self.window_size).to(self.device)
                    seq = F.one_hot(seq,num_classes=self.num_classes).float()
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = self.model(seq, self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        TP += 1
                        break

        elapsed_time = time.time() - start_time
        print('elapsed_time: {:.3f}s'.format(elapsed_time))
        # Compute precision, recall and F1-measure
        FN = len(test_abnormal_loader) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
        print('Finished Predicting')
    
    def save_model(self, model_pth):
        torch.save(self.model, model_pth)

    def load_model(self, model_pth):
        self.model = torch.load(model_pth)


class LSTM_onehot(nn.Module):
    def __init__(self, hidden_size, num_layers, num_keys):
        super(LSTM_onehot, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_keys, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


