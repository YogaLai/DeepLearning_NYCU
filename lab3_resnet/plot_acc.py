import sys
import matplotlib.pyplot as plt

if len(sys.argv)!=2:
    print('Please input the record text file')
    exit()
filename=sys.argv[1]
f=open(filename)
for line in f:
    line=line[1:-2]
    line=line.replace(' ', '')
    acc_list=line.split(',')
for i in range(len(acc_list)):
    acc_list[i]=float(acc_list[i])

plt.plot(range(len(acc_list)),acc_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(('result/accuracy.png'))
f.close()