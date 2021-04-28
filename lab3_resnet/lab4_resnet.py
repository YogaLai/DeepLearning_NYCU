from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
import matplotlib.pyplot as plt
from dataloader import RetinopathyLoader
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type",
    help="Choose resnet18 or resnet50", type=str, default='18')
args=parser.parse_args()
model_type=args.type

def train(model,pretrain,data,correct):
    if pretrain:
        optim=pretrain_optimizer
    else:
        optim=optimizer
    optim.zero_grad()
    # optimizer.zero_grad()
    inputs,labels=data
    inputs,labels=inputs.to(device),labels.to(device)
    outputs=model(inputs)
    _, predict = torch.max(outputs.data, 1)
    correct+=(predict==labels).sum().item()

    loss = criterion(outputs, labels)
    loss.backward()
    optim.step()
    # optimizer.step()

    return loss.item(), correct

device = torch.cuda.current_device()
print('device:',device)

#Parameter
epochs=10
lr=0.001
batch_size=64
momentum=0.9
criterion=nn.CrossEntropyLoss()
num_classes=5
weight_decay=0.0005

train_loader=RetinopathyLoader('data/','train')
trainset_size=len(train_loader)
train_loader=DataLoader(train_loader,batch_size=batch_size,shuffle=True)
test_loader=RetinopathyLoader('data/','test')
testset_size=len(test_loader)
test_loader=DataLoader(test_loader,batch_size=batch_size,shuffle=True)

if model_type=='18':
    model = models.resnet18(pretrained=args.pretrain)
    num_neurons = self.model.fc.in_features
    model.fc=nn.Linear(num_neurons, num_classes)
    model=model.cuda()

    # model_pretraining = models.resnet18(pretrained=True)
    # num_neurons = self.model_pretraining.fc.in_features
    # model_pretraining.fc=nn.Linear(num_neurons, num_classes)
    # model_pretraining=model_pretraining.cuda()
    for param in model_pretraining.parameters():
        param.requires_grad=False
else:
    model = models.resnet50()
    model.fc=nn.Linear(512, num_classes)
    model=model.cuda()

    model_pretraining = models.resnet50(pretrained=True)
    model_pretraining.fc=nn.Linear(512, num_classes)
    model_pretraining=model_pretraining.cuda()

optimizer=optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
pretrain_optimizer=optim.SGD(model.parameters(),lr=0.0005,momentum=momentum,weight_decay=weight_decay)

# ignored_params = list(map(id, resnet18_pretraining.fc.parameters()))
# base_params = filter(lambda p: id(p) not in ignored_params, resnet18_pretraining.parameters())

# # 对不同参数设置不同的学习率
# params_list = [{'params': base_params, 'lr': 0.0001},]
# params_list.append({'params': resnet18_pretraining.fc.parameters(), 'lr': 0.001})
# optimizer_pretraining=optim.SGD(params_list,lr=lr/1000,momentum=0.9,weight_decay=0.005)

#Train
train_acc=[]
pretrain_train_acc=[]
test_acc=[]
pretrain_test_acc=[]
test_label=[]
test_predict=[]
epoch_loss=[]
pretrain_epoch_loss=[]
for epoch in range(epochs):
    # Training
    correct=0
    pretrain_correct=0
    running_loss=0.0 
    pretrain_running_loss=0.0
    model.train()
    model_pretraining.train()    
    for times,data in enumerate(train_loader):
        loss, correct = train(model,optimizer,data,correct)
        running_loss += loss
        print('[%d/%d] [%d/%d] loss: %.3f' % (epoch+1, epochs, times, trainset_size/batch_size, loss))           
        
        loss, pretrain_correct = train(model,pretrain_optimizer,data,pretrain_correct)
        pretrain_running_loss += loss
        print('Pretraing [%d/%d] [%d/%d] loss: %.3f' % (epoch+1, epochs, times, trainset_size/batch_size, loss))

    epoch_loss.append(running_loss / trainset_size)
    pretrain_epoch_loss.append(pretrain_running_loss / trainset_size)
    train_acc.append(correct/trainset_size*100)
    pretrain_train_acc.append(pretrain_correct/trainset_size*100)
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss[-1], train_acc[-1]))
    print('PreTrain Loss: {:.4f} Acc: {:.4f}'.format(pretrain_epoch_loss[-1], pretrain_train_acc[-1]))

    
    # Testing
    print('Testing...')
    correct=0.0
    pretrain_correct=0.0
    model.eval()
    model_pretraining.eval()
    with torch.no_grad():
      for data in test_loader:
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        _, predict = torch.max(outputs.data, 1)
        correct+=(predict==labels).sum().item()
        
        outputs=model_pretraining(inputs)
        _, predict = torch.max(outputs.data, 1)
        if epoch==epochs-1:
          if len(test_label)==0:
            test_label=labels
            test_predict=predict
          else:
            test_label=torch.cat((test_label,labels),dim=0)
            test_predict=torch.cat((test_predict,predict),dim=0)
        pretrain_correct+=(predict==labels).sum().item()

    test_acc.append(correct/testset_size*100)
    pretrain_test_acc.append(pretrain_correct/testset_size*100)
    print('Test acc {:.2f}%' .format(test_acc[-1]))
    print('Pretrain test acc {:.2f}%' .format(pretrain_test_acc[-1]))
    

# 畫圖
plt.figure()
plt.plot(range(epochs), epoch_loss, color='b', label='w/o pretrain')
plt.plot(range(epochs), pretrain_epoch_loss, color='r', label='w pretrain')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('result/loss.png')

plt.figure()
plt.plot(range(epochs),train_acc,color='b',label='Train(w/o pretraining)',marker='.')
plt.plot(range(epochs),pretrain_train_acc,color='r',label='Train(with pretraining)')
plt.plot(range(epochs),test_acc,color='y',label='Test(w/o pretraining)',marker='.')
plt.plot(range(epochs),pretrain_test_acc,color='g',label='Test(with pretraining)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('result/accuracy.png')

# Confusion matrix
stacked = torch.stack(
    (
        test_label
        ,test_predict
    )
    ,dim=1
)
cm = torch.zeros(num_classes,num_classes)
for item in stacked:
    label, predict = item.tolist()
    cm[label, predict] = cm[label, predict] + 1
print(cm.sum(axis=1)[:,np.newaxis].shape)
cm=cm/cm.sum(axis=1)[:,np.newaxis]
# for i in range(num_classes):
#  cm[i]=cm[i]/cm[i].sum()

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
# tick_marks = np.arange(num_classes)
# plt.xticks(tick_marks, num_classes, rotation=45)
# plt.yticks(tick_marks, num_classes)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    plt.text(j, i, format(cm[i, j],'.2f'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalize confusion matrix')
plt.savefig('result/confusion_matrix')
