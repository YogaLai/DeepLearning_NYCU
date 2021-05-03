import sys
import matplotlib.pyplot as plt

if len(sys.argv)!=2:
    print('Please input the record type')
    exit()

model_type=sys.argv[1]

def getAcc(filename):
    f=open('result/' + filename)
    acc=[]
    for line in f:
        line=line[1:-2]
        line=line.replace(' ', '')
        acc=line.split(',')
    for i in range(len(acc)):
        acc[i]=float(acc[i])
    return acc

if model_type == '18':
    train_acc = getAcc('18_train_acc.txt')
    train_acc_pretrain = getAcc('18_pretrain_train_acc.txt')
    test_acc = getAcc('18_test_acc.txt')
    test_acc_pretrain = getAcc('18_pretrain_test_acc.txt')
else:
    train_acc=getAcc('50_train_acc.txt')
    train_acc_pretrain=getAcc('50_pretrain_train_acc.txt')
    test_acc = getAcc('50_test_acc.txt')
    test_acc_pretrain = getAcc('50_pretrain_test_acc.txt')

epochs=range(len(train_acc))
plt.title('Result Comparision(ResNet' + model_type + ')')
plt.plot(epochs,train_acc,color='b',label='Train(w/o pretraining')
plt.plot(epochs,test_acc,color='r',label='Test(w/o pretraining')
plt.plot(epochs,train_acc_pretrain,color='y',label='Train(with pretraining)',marker='.')
plt.plot(epochs,test_acc_pretrain,color='g',label='Test(with pretraining)',marker='.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(('result/accuracy.png'))

# show highest accuracy
print('The higest testing accuracy (ResNet' + model_type + ')')
print(max(test_acc_pretrain))
