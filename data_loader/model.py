import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import joblib
from lightgbm import LGBMRegressor, plot_importance
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import params
from sklearn.utils import shuffle
from tqdm import tqdm
import time


# ANN module
import torch
from torch import nn, optim   
import torch.nn.functional as F    


data = params.DATA_SET
# load
with open(data, 'rb') as f:
    vector_dataset = pickle.load(f)




X_dataset = vector_dataset.iloc[:,:params.ANALYSIS_DAY].reset_index(drop=True)
# # 5일씩 묶어보자
# for idx,i in enumerate(range(0,params.ANALYSIS_DAY,5)):
#     X_dataset[f"new_{idx}"] = X_dataset[i] + X_dataset[i+1]+ X_dataset[i+2] +X_dataset[i+3] +X_dataset[i+4]
# X_dataset = X_dataset.iloc[:,-9:].reset_index(drop=True)

Y_dataset = vector_dataset.iloc[:,params.ANALYSIS_DAY*2:].reset_index(drop=True)
print(Y_dataset)
# random_seed 설정
SEED = 1

# Train Vaild Test split
X_train, X_vaild, Y_train, Y_vaild = train_test_split(X_dataset, Y_dataset, test_size=0.1, random_state=SEED, shuffle=True)

X_train = torch.Tensor(X_train.values)
Y_train = torch.Tensor(Y_train.values)
print(X_train.shape)
print(Y_train.shape)
X_vaild = torch.Tensor(X_vaild.values)
Y_vaild = torch.Tensor(Y_vaild.values)
print(X_vaild.shape)
print(Y_vaild.shape)
# X_test = torch.Tensor(X_test.values)
# Y_test = torch.Tensor(Y_test.values)
# print(X_test.shape)
# print(Y_test.shape)


# model Hyperparams
batch_size = 15000
hidden_dim = 64
epochs = 50
dropout = 0.5

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# defining utility class
# by defining this, you only have to write "for loop" to load minibatch data
class DataLoader(object):
    def __init__(self, x, y, batch_size=batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_idx = 0
        self.data_size = x.shape[0]
        if self.shuffle:
            self.reset()
    
    def reset(self):
        self.x, self.y = shuffle(self.x, self.y)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start_idx >= self.data_size:
            if self.shuffle:
                self.reset()
            self.start_idx = 0
            raise StopIteration
    
        batch_x = self.x[self.start_idx:self.start_idx+self.batch_size]
        batch_y = self.y[self.start_idx:self.start_idx+self.batch_size]

        batch_x = torch.tensor(batch_x, dtype=torch.float, device=device)
        batch_y = torch.tensor(batch_y, dtype=torch.float, device=device)

        self.start_idx += self.batch_size

        return (batch_x,batch_y)

#defining MLP model
#generally out_dim is more than 1, but this model only allows 1.
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super(MLP, self).__init__()
        
        assert out_dim==1, 'out_dim must be 1'
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 32)
        self.linear3 = nn.Linear(32, 1)
        self.drop2D = nn.Dropout(p=dropout, inplace=False)  
        self.LeakyReLU = nn.LeakyReLU(0.1)
    
        
    
    def forward(self, x):
        
        x = self.LeakyReLU(self.linear1(x))
        x = self.drop2D(x)
        x = torch.sigmoid(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = x.squeeze(1)
        return x


#instantiate model
mlp = MLP(X_train.shape[1], hidden_dim, 1).to(device)
# Adam -->SGD
optimizer = optim.SGD(mlp.parameters(), lr=0.001, weight_decay=0.00001,momentum=0.9,nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)

train_dataloader = DataLoader(X_train, Y_train, batch_size=batch_size)
valid_dataloader = DataLoader(X_vaild, Y_vaild, batch_size=batch_size)  


#this model learns to minimize MAE
def mae_loss(y_pred, y_true):
    mae = torch.abs(y_true - y_pred).mean()
    return mae
# create a function (this my favorite choice)
def RMSELoss(yhat,y):
    rmse = torch.sqrt(torch.mean((yhat-y)**2))
    return rmse




#to plot loss curve after training
valid_losses = []

for epoch in range(epochs):
    start_time = time.time()
    mlp.train()
    num_batch = train_dataloader.data_size // train_dataloader.batch_size + 1
    
    for batch_id, (batch_x, batch_y) in enumerate(train_dataloader):
        
        y_pred = mlp(batch_x)

        loss = mae_loss(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = elapsed_time - 60 * elapsed_min

        print('\rEpoch:{} Batch:{}/{} Loss:{:.4f} Time:{}m{:.2f}s'.format(epoch + 1, batch_id, 
                                                                          num_batch, loss.item(),
                                                                          elapsed_min, elapsed_sec), end='')
    scheduler.step()
    print()
    mlp.eval()
    valid_loss = 0
    best_loss = np.inf
    num_batch = valid_dataloader.data_size // valid_dataloader.batch_size + 1
    
    for batch_id, (batch_x, batch_y) in enumerate(valid_dataloader):
    
        y_pred = mlp(batch_x)
        loss = mae_loss(y_pred, batch_y)
        valid_loss += loss.item()
    
    valid_loss /= num_batch
    valid_losses.append(valid_loss)
    
    #save model when validation loss is minimum
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(mlp.state_dict(), f'../model/mlp_model_batch_{batch_size}_best_{best_loss:.6f}.model')  
    
    print('Valid Loss:{:.4f}'.format(valid_loss))
    

#plot validation loss curve, this may help to notice overfitting
plt.figure(figsize=(16,5))
plt.ylim(0,max(valid_losses)+0.02)
plt.plot(valid_losses)
print('minimum validation loss is {:.4f}'.format(min(valid_losses)))