
# coding: utf-8

# In the last chapter, we used the trained neural network to predict labels for test data. Now, we use the train dataset to make the neural network learn parameters such as weight and bias.
# 
# ### Non machine learning algorithms:  
# data -> alg -> answer
# 
# ### Traditional machine learning algorithms (e.g. SVM, KNN):  
# Use human prepared feature vectors like SIFT, SURF, HOG.  
# data -> feature vector -> answer
# 
# ### Deep learning:  
# data -> deep learning -> answer

# ## Loss Function

# In[1]:

import numpy as np


# In[2]:

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)


# In[3]:

def cross_entropy_error(y,t): # the higher the prob for the correct label is the closer to 0 the product becomes
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# In[4]:

t = np.array([0,0,0,1]) # answer
y1 = np.array([0,0.1,0.2,0.7]) # good prediction
y2 = np.array([0.8,0.1,0,0.1]) # bad prediction


# In[5]:

mean_squared_error(y1,t),mean_squared_error(y2,t)


# In[6]:

cross_entropy_error(y1,t),cross_entropy_error(y2,t)


# ## Minibatch
# Loss function can easily be extended for n data points by adding individual loss functions and taking the average. The problem is that with big data, this calculation will take forever. Hence, we randomly sample a portion of the dataset. This is called a minibatch.

# In[7]:

from ch5_predict_mnist import load_mnist


# In[8]:

(x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)


# In[9]:

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # randomly select batch_size samples
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# In[10]:

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size # for a batch, only get the prob for the label with the correct prob


# ## Derivates

# In[11]:

def numerical_diff_bad(f,x):
    h = 10e-50
    return (f(x+h)-f(x))/h


# We can improve this by using 
# 1. a large enough h to avoid rounding error.
# 2. use f(x+h)-f(x-h) instead of f(x+h)-f(x) (central diff)

# In[12]:

def numerical_diff(f,x):
    h = 10e-4
    return (f(x+h)-f(x-h)) / 2*h


# ## Gradient
# Partial derivatives for every variable

# In[13]:

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        
    return grad


# In[14]:

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


# In[15]:

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


# In[16]:

def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


# In[47]:

from mpl_toolkits.mplot3d import Axes3D
x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(function_2, np.array([X, Y]) )

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()


# ## Gradient Descent

# In[17]:

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        print(x)
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


# In[18]:

def function_2(x):
    return x[0]**2 + x[1]**2


# In[19]:

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()


# In[ ]:

init_x = np.array([-3.0, 4.0])    

lr = 1 # lr too big
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()


# In[20]:

init_x = np.array([-3.0, 4.0])    

lr = 0.01 # lr too small
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()


# ## Train a Two Layer Neural Network using SGD
# Let's first review the steps for learning.  
# 0. The goal is to learn weights and biases based on train data.  
# 1. (Minibatch) Randomly sample data from train dataset.  
# 2. (Gradient) To reduce the loss function of minibatch, calculate the gradient for each parameter.  
# 3. (Update) Update the parameters based on gradient.
# 4. Repeat step 2,3,4.
# 
# This method of sampling a minibatch to update the parameters is called a stochastic gradient descent.

# In[21]:

from utils import *

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # initialize weights
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:input data, t:train data
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


# In[22]:

(x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)


# In[23]:

train_loss_list = []


# In[40]:

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)
train_loss_list = []
train_acc_list = []
test_acc_list = []


# In[41]:

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# In[42]:

for i in range(iters_num):
    if i % 1000 == 0:
        print(i)
    # minibatch
    batch_mask = np.random.choice(train_size, batch_size) # randomly sample indices
    x_batch = x_train[batch_mask] # only get the indices randomly sampled
    t_batch = t_train[batch_mask]
    
    # gradient
    grad = network.gradient(x_batch, t_batch)
    
    # update
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # record loss function value
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# In[44]:

import matplotlib.pylab as plt

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# In[47]:

x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='train acc')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc='lower right')
plt.show()


# In[ ]:



