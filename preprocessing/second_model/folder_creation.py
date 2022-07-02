import os
import string

globalfile = 'E:/Documents/Github/Express-U/data/Live_dataset'

a = list(string.ascii_uppercase)
#train
for i in range(1,10):
  os.mkdir(f'{globalfile}/train/{i}')

for i in range(26):
  os.mkdir(f'{globalfile}/train/{a[i]}')

#test
for i in range(1,10):
  os.mkdir(f'{globalfile}/test/{i}')

for i in range(26):
  os.mkdir(f'{globalfile}/test/{a[i]}')


