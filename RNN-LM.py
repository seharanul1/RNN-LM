
# coding: utf-8

# In[68]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import pickle
 


# In[69]:

data = pickle.load(open("datasets/data.pickle","rb"))
data_train = data['train']
data_val = data['valid']
data_test = data['test']


# In[70]:

data_train[:200]


# In[ ]:




# In[71]:

dtype = torch.FloatTensor # the CPU datatype


# In[72]:

len(data)


# In[73]:

lr = 1.0
batch_size = 20
num_epochs = 1
clip = 5


# In[74]:

train_batch=[] #get batch 함수
#lstm-char-cnn-tensorflow/models at master · carpedm20/lstm-char-cnn-tensorflow · GitHub


# In[75]:

char_vec = torch.randn(15,26).type(dtype)
#char_vec = np.random.randn(15,26)
def charToIndex(c) :
    return ord(c)-ord('a')
def charToVec(c) :
    return char_vec[:,charToIndex(c)]

#print(char_vec,"!")
#print(char_vec[:,2],"!")
#print(charToVec('c'),"!")


# In[76]:

def wordToVec(w):
    #zero = np.zeros(15)
    zero = torch.zeros(15)
    max_size = 21
    out = torch.stack([charToVec(c) for c in w]+[zero for _ in range(max_size-len(w))],dim=1)
    mask = [1 for _ in range(len(w))] + [0 for _ in range(max_size-len(w))]
    #return (out, mask)
    return out
#wordToVec('abd')


# In[77]:

def batchToVec(x, batch_size):
    out = torch.FloatTensor(batch_size, 1, 15,21)
    for i in range(batch_size):
        out[i][0] = wordToVec(x[i])
    return out
print(batchToVec(['abc','deewef'], 2).shape)


# In[ ]:




# In[78]:

#x는 N, 21일 것
#c_x는 N, 15, 21
#N개의 c_w에 15씩 21개 들어있음


# In[212]:

batch_size = 2
x = ['apple','an']
out= batchToVec(x, len(x))
print(out.shape)
out = Variable(out)


# In[213]:

class charCNN(nn.Module):
    def __init__(self):
        super(charCNN, self).__init__()
        self.filter_size = [1,2,3,4,5,6]
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 25*w, kernel_size=(15,w)) for w in self.filter_size]
        )
        self.tanh = nn.Tanh()
        
    def forward(self, x):    
        out = None
        for conv in self.convs:
            y = self.tanh(conv(x))
            #print(y.data.shape)
            y = torch.squeeze(y)
            print(y.data.shape)
            #print(y.data[0])
            #print("!", y.data[0][0])           
            
            #max_y, _ = torch.max(y.data[0], dim=1) 
            #print(max_y)
            max_y, _ = torch.max(y, dim=2)
            #print(max_y)
            if out is not None:
                print(out.data.shape, max_y.data.shape)
                #out = torch.stack((out,max_y), dim=1)
                out = torch.cat((out,max_y),dim=1)
      
            else:
                out = max_y
            print("outsize :", out.data.shape)
            #print("size", out[0].data.shape)
            #print("out:\n", out)
            #print("!!!!!!!!!!!!!!!!!!")
        return out


# In[214]:

"""a = Variable(torch.randn(2,3))
b = Variable(torch.randn(2,5))

print(a.data.shape, b.data.shape)
torch.cat((a,b), dim=1)"""
filter_size = [1,2,3,4,5,6]
print(np.sum([25*i for i in filter_size]))


# In[215]:

A = charCNN()
out = A(out)


# In[190]:

class Highway(nn.Module):
    def __init__(self, num_layers):
        super(Highway, self).__init__()
        
        input_size = np.sum([25*i for i in filter_size])
    
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (relu(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | 
            G and Q is affine transformation,
            f is non-linear transformation,
            σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = F.relu(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


# In[152]:

class LSTM(self):
    def __init__(self):
        #
    def forward(self):
        #
    def backward(self):
        #35step 씩


# In[18]:




# In[ ]:

class Model(nn.Module):
    def _init__(self):
        super(Model, self).__init()
        
        self.charCNN= charCNN()
        self.highway = Highway(num_layers=1)
        
        self.LSTM = nn.LSTM(input_size = 0,hidden_size=300,num_layers=2,dropout=0.5)
        
    def forward(self, x):
        out = self.batchToVec(x, batch_size)
        out = self.charCNN(out)  
        out = self.highway(out)
        out = self.LSTM(out)
 
        return out


#http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


# In[ ]:

model = Model()
model = model.type(dtype)

criterion = nn.CrossEntropyLoss().type(dtype)
#optimizer = optim.RMSprop(model.parameters(), lr=lr)


# To make sure you're doing the right thing, use the following tool to check the dimensionality of your output (it should be 64 x 10, since our batches have size 64 and the output of the final affine layer should be 10, corresponding to our 10 classes):

# In[ ]:

## Now we're going to feed a random batch into the model you defined and make sure the output is the right size
x = torch.randn(64, 3, 32, 32).type(dtype)
x_var = Variable(x.type(dtype)) # Construct a PyTorch Variable out of your input data
ans = model(x_var)        # Feed it through the model! 

# Check to make sure what comes out of your model
# is the right dimensionality... this should be True
# if you've done everything correctly
np.array_equal(np.array(ans.size()), np.array([64, 10]))       


# ### Train 

# In[ ]:

def train(model, loss_fn, optimizer, num_epochs = num_epochs):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        
        model.train()
        
        for t, (x, y) in enumerate(data_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)
            loss = criterion(scores, y_var)
            
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            
            #gradient clippings
            nn.utils.clip_grad_norm(model.parameters(), clip, norm_type=2)
            
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


# In[ ]:

torch.manual_seed(12345)
model.apply(reset)
train(fixed_model_gpu, loss_fn, optimizer, num_epochs=1)
check_accuracy(fixed_model_gpu, loader_val)


# ### Don't forget the validation set!
# 
# And note that you can use the check_accuracy function to evaluate on either the test set or the validation set, by passing either **loader_test** or **loader_val** as the second argument to check_accuracy. You should not touch the test set until you have finished your architecture and hyperparameter tuning, and only run the test set once at the end to report a final value. 

# In[ ]:

# Train your model here, and make sure the output of this cell is the accuracy of your best model on the 
# train, val, and test sets. Here's some code to get you started. The output of this cell should be the training
# and validation accuracy on your best model (measured by validation accuracy).

model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=1),
                nn.ReLU(inplace=True),
    
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                Flatten(), # see above for explanation
                nn.Linear(32*13*13, 1024), # affine layer
                nn.ReLU(inplace=True),
    
                nn.Linear(1024, 10)    
)

model = model.type(dtype)

loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


train(model, loss_fn, optimizer, num_epochs=1)
check_accuracy(model, loader_val)


# In[ ]:

best_model = None
check_accuracy(best_model, loader_test)

