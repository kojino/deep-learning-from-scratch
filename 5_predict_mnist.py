
# coding: utf-8

# ## Import MNIST dataset

# In[3]:

import urllib.request
import os.path
import gzip
import pickle
import sys,os
import numpy as np


# In[4]:

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.getcwd()
save_file = dataset_dir + "/mnist.pkl"


# In[5]:

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


# In[6]:

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


# In[7]:

def download_mnist():
    for v in key_file.values():
       _download(v)


# In[8]:

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels


# In[9]:

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data


# In[10]:

def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset


# In[11]:

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


# In[12]:

def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T


# In[13]:

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) f


# In[25]:

from PIL import Image
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# In[26]:

(x_train,t_train), (x_test,t_test) = load_mnist(flatten=True,normalize=False)


# In[27]:

print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)


# In[28]:

img = x_train[0]
label = t_train[0]
print(label)


# In[29]:

print(img.shape)


# In[30]:

img = img.reshape(28,28)
print(img.shape)


# In[31]:

img_show(img)


# ## Test the dataset
# In the later chapter, we'll cover how to train the dataset. Here, we'll assume that the trained neural network is already in our hands, and use it to classify a newly provided test dataset.

# In[32]:

def get_data():
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test, t_test


# In[34]:

def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network


# In[36]:

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[38]:

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp = np.sum(exp_a)
    return exp_a/sum_exp


# In[46]:

def predict(network,x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    
    return y


# In[47]:

x, t = get_data()
network = init_network()


# In[48]:

accuracy_cnt = 0
for i in range(len(x)): # predict the label for each number
    y = predict(network, x[i])
    p = np.argmax(y) # get the label with the highest prob
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy: " + str(float(accuracy_cnt)/len(x)))


# This can be optimized by processing using batches (e.g. process 100 inputs at a time)

# In[49]:

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size): # predict the label for each batch
    x_batch = x[i:i+batch_size] # get 100 data at once
    y = predict(network, x_batch)
    p = np.argmax(y, axis=1) # get the index with the highest prob for each row.
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
print("Accuracy: " + str(float(accuracy_cnt)/len(x)))


# axis=1: by row, axis=0: by column

# In[55]:

x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
np.argmax(x, axis=1), np.argmax(x, axis=0)

