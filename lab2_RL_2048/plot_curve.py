import matplotlib.pyplot as plt

file=open('score_record.txt')
episode=[]
score=[]
for line in file:
    line=line.split(' ')
    episode.append(int(line[0]))
    score.append(int(line[1].replace('\n','')))

plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Learning Curve')
plt.plot(episode,score)
plt.savefig('learning curve.png')