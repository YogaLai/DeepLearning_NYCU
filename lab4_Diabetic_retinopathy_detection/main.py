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
parser.add_argument("--pretrain",
    help="w or w/o pretrain", action="store_true")
args=parser.parse_args()
model_type=args.type

train_acc=[]
pretrain_train_acc=[]
test_acc=[]
pretrain_test_acc=[]
test_label=[]
test_predict=[]
epoch_loss=[]
pretrain_epoch_loss=[]

def train(model,optimizer,epochs,confusion=False):
    global test_label, test_predict, train_acc, test_acc
    for epoch in range(epochs):
        # Training
        correct=0
        running_loss=0.0 
        model.train()
        for times,data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            _, predict = torch.max(outputs.data, 1)
            correct+=(predict==labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('[%d/%d] [%d/%d] loss: %.3f' % (epoch+1, epochs, times, trainset_size/batch_size, loss.item()))           

        epoch_loss.append(running_loss / trainset_size)
        train_acc.append(correct/trainset_size*100)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss[-1], train_acc[-1]))

        print('Testing...')
        correct=0.0
        with torch.no_grad():
            model.eval()
            for data in test_loader:
                inputs,labels=data
                inputs,labels=inputs.to(device),labels.to(device)
                outputs=model(inputs)
                _, predict = torch.max(outputs.data, 1)
                correct+=(predict==labels).sum().item()
                
                if epoch==epochs-1 and confusion:
                    if len(test_label)==0:
                        test_label=labels
                        test_predict=predict
                    else:
                        test_label=torch.cat((test_label,labels),dim=0)
                        test_predict=torch.cat((test_predict,predict),dim=0)

        test_acc.append(correct/testset_size*100)
        print('Test acc {:.2f}%' .format(test_acc[-1]))

device = torch.cuda.current_device()
print('device:',device)

#Parameter
epochs=7
feat_ext_epochs=3
lr=5e-4
batch_size=12
momentum=0.9
criterion=nn.CrossEntropyLoss()
num_classes=5
weight_decay=5e-4

train_loader=RetinopathyLoader('data/','train')
trainset_size=len(train_loader)
train_loader=DataLoader(train_loader,batch_size=batch_size,shuffle=True)
test_loader=RetinopathyLoader('data/','test')
testset_size=len(test_loader)
test_loader=DataLoader(test_loader,batch_size=batch_size,shuffle=False)

if model_type=='18':
    model = models.resnet18(pretrained=args.pretrain)
else:
    model = models.resnet50(pretrained=args.pretrain)

if args.pretrain:
    for param in model.parameters():
        param.requires_grad=False
num_neurons = model.fc.in_features
model.fc=nn.Linear(num_neurons, num_classes)
model=model.cuda()

if args.pretrain:
    params_to_update=[]
    for name,param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer=optim.SGD(params_to_update,lr=lr,momentum=momentum,weight_decay=weight_decay)
else:
    optimizer=optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)

# ignored_params = list(map(id, model.fc.parameters()))
# base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

# # 对不同参数设置不同的学习率
# params_list = [{'params': base_params, 'lr': 0.0001},]
# params_list.append({'params': model.fc.parameters(), 'lr': 0.001})
# optimizer=optim.SGD(params_list,lr=lr,momentum=momentum,weight_decay=weight_decay)
# train(model, optimizer, epochs, confusion=True)

if args.pretrain:
    # Only train output linear layer, don't train feature extraction layers
    train(model,optimizer,feat_ext_epochs)
    # finetuning
    for param in model.parameters():
        param.requires_grad=True
    optimizer=optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    train(model,optimizer,epochs-feat_ext_epochs,confusion=True)
    torch.save(model, 'model/' + model_type+'_pretrain.pth')
else:
    train(model,optimizer,epochs,confusion=True)
    torch.save(model, 'model/' + model_type+'.pth')

plt.figure()
plt.plot(range(epochs), epoch_loss, color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('result/loss.png')

train_record=None
# test_record=None
if not args.pretrain:
    train_record=open('result/'+model_type+'_train_acc.txt','w')
    test_record=open('result/'+model_type+'_test_acc.txt','w')
else:
    train_record=open('result/'+model_type+'_pretrain_train_acc.txt','w')
    test_record=open('result/'+model_type+'_pretrain_test_acc.txt','w')

print(train_acc,file=train_record)
print(test_acc,file=test_record)

'''
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
'''

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
