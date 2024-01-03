import random
import sys

src=sys.argv[1]
des=sys.argv[2]
n=sys.argv[3]
li = []
with open(src, "r") as f:
  for line in f:
    li.append(line)

random.shuffle(li)


f1 = open(des, "a")
for i in range(int(n)):
  f1.write(li[i])

