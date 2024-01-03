import os

n=2000
m=20
if os.path.exists("f"):
  os.remove("f")
os.mknod("f")
f1 = open("f", "a")
for i in range(n):
  line = str(i)
  for j in range(m):
    line += " 0"
  f1.write(line + '\n')



