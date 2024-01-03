#encoding=utf-8
import os
import sys
path=sys.argv[1]
des = sys.argv[2]
def is_en(w):
    if 'a'<=w<='z' or 'A'<=w<='Z':
        return True
if os.path.exists(des):
    os.remove(des)
    os.mknod(des)
with open(path, "r") as f, open (des, "a") as f1:
    for line in f:
        str = []
        line = line.splitlines()[0]
        i = 0
        while i < len(line):
           if is_en(line[i]):
              j = i
              en = ""
              while j < len(line) and is_en(line[j]):
                en += line[j]
                j += 1
              str.append(en)
              i = j    
           else:
              if line[i] == ' ':
                 i += 1
                 continue
              str.append(line[i:i+1])
              #print(line[i:i+3])
              i += 1
                
        #print(str)
        str = " ".join(str)
        str += '\n'
        f1.write(str)

