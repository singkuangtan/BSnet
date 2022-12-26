import re
import matplotlib.pyplot as plt
import numpy as np

#open text file in read mode
text_file = open("normal.txt", "r")
#read whole file to a string
data = text_file.read()
#close file
text_file.close()
 
#print(data)


str = data
match = re.findall(r'val\_accuracy\: (\d*\.\d*)', str)
match=np.array(match)
print(match)
plt.plot(match.astype(float), label='Normal Network Validation Accuracy')
#plt.show()

str = data
match = re.findall(r' accuracy\: (\d*\.\d*)', str)
match=np.array(match)
match=match[:-1]
print(match)
plt.plot(match.astype(float), label='Normal Network Training Accuracy')
#plt.show()
  
#'''
#open text file in read mode
text_file = open("bsnet.txt", "r")
#read whole file to a string
data = text_file.read()
#close file
text_file.close()
 
#print(data)


str = data
match = re.findall(r'val\_accuracy\: (\d*\.\d*)', str)
match=np.array(match)
print(match)
plt.plot(match.astype(float), label='BSnet Validation Accuracy')
#plt.show()

str = data
match = re.findall(r' accuracy\: (\d*\.\d*)', str)
match=np.array(match)
match=match[:-1]
print(match)
plt.plot(match.astype(float), label='BSnet Training Accuracy')

plt.legend()
plt.show()
#'''