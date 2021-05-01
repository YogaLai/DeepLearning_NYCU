import sys
import matplotlib.pyplot as plt

if len(sys.argv)!=3:
    print('Please input the record text file')
    exit()
train_record=sys.argv[1]
test_record=sys.argv[2]
f=open(train_record)
for line in f:
    line=line[1:-2]
    line=line.replace(' ', '')
    train_acc=line.split(',')
for i in range(len(train_acc)):
    train_acc[i]=float(train_acc[i])

f=open(test_record)
for line in f:
    line=line[1:-2]
    line=line.replace(' ', '')
    test_acc=line.split(',')
for i in range(len(test_acc)):
    test_acc[i]=float(test_acc[i])
f.close()

plt.plot(range(len(train_acc)),train_acc,color='b',label=train_record.split('.')[0])
plt.plot(range(len(test_acc)),test_acc,color='r',label=test_record.split('.')[0])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(('result/accuracy.png'))