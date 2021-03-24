import numpy as np
from util import *
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--type",
    help="Choose the data type\n(0) linear \n(1) XOR", type=int, default=0)
parser.add_argument("--bias",
    help="Network add bias or not", action="store_true")
parser.add_argument("-M",
    help="Add momentum", action="store_true")
args=parser.parse_args()

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x,1.0-x)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def derivative_tanh(x):
    return 1.-tanh(x)**2

def err_loss(pred,labels):
    return np.mean((pred-labels)**2)

def derivative_loss(pred,labels):
    return (pred-labels)*(2/pred.shape[0])

class Layer():
    def __init__(self,input_dim,output_dim,bias):
        self.bias=bias
        if self.bias:
            self.w = np.random.normal(0, 1, (input_dim+1, output_dim)) # add bias
        else:
            self.w = np.random.normal(0, 1, (input_dim, output_dim))
        self.last_graident = 0

    def forward(self,x):
        if self.bias:
            self.x = np.append(x, np.ones((x.shape[0],1)), axis=1) # add bias
        else:
            self.x=x
        self.z=tanh(self.x@self.w)
        return self.z

    def backprop(self,dloss):
        self.backward_gradient=np.multiply(dloss, derivative_tanh(self.z)) # dC/dz = dC/dy(dloss) * dy/dz(d_sigmoid)
        if self.bias:
            return self.backward_gradient @ self.w[:-1].T # w[:-1] remove bias column
        else:
            return self.backward_gradient @ self.w.T # w[:-1] remove bias column

    def optimize(self,lr):  # Gradient Descent w-=lr*dC/dw dC/dw = dC/dz(backward_gradient) * dz/dw(x)
        gradient=self.x.T @ self.backward_gradient
        if args.M:
            self.w = self.w - lr * gradient + 0.9 * self.last_graident  # 0.9 is momentum
            self.last_graident = gradient
        else:
            self.w -= lr * gradient   

class Net():
    # Initialize a network
    def __init__(self,layer_setting,bias,lr=0.01):
        self.lr=lr
        self.layers=[]
        for input_dim,output_dim in zip(layer_setting[0],layer_setting[1]):
            self.layers.append(Layer(input_dim,output_dim,bias))

    def forward(self,inputs):
        y=inputs
        for layer in self.layers:
            y=layer.forward(y)
        return y
    
    def backprop(self,pred,labels):
        dloss=derivative_loss(pred,labels)
        for layer in self.layers[::-1]: # reverse order
            dloss=layer.backprop(dloss)
    
    def optimize(self):
        for layer in self.layers:
            layer.optimize(self.lr)

def main():
    data_type={
        'Linear':0,
        'XOR':1
    }
    if args.type==data_type['Linear']:
        inputs,labels=generate_linear()
        tag='Linear'
    else:
        inputs,labels=generate_XOR()
        tag='XOR'

    layer_setting=[[2,10,10],[10,10,1]]
    net=Net(layer_setting,args.bias,lr=1)
    epochs=10000
    writer=SummaryWriter()
    for epoch in range(epochs):
        pred=net.forward(inputs)
        loss=err_loss(pred,labels)
        net.backprop(pred,labels)
        net.optimize()
        if epoch % 10 ==0:
            print('epoch %d loss :%f' % (epoch,loss))
            writer.add_scalar(tag+' loss',loss,epoch)
        if loss<0.03:
            print('Converge')
            break

    pred=np.round(pred)
    t=np.count_nonzero(pred==labels)
    show_result(inputs,labels,pred)
    print('Accuracy: %.2f%%' % (np.count_nonzero(pred==labels)/len(labels)*100))

if __name__=='__main__':
    main()