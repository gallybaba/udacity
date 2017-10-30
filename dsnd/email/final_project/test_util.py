import pickle
import sys
import numpy as np
import pandas as pd

#original = 'final_project/final_project_dataset.pkl'
modified = 'final_project/final_project_dataset_modified.pkl'

#originalf = open(original, 'rb')
#modifiedf = open(modified, 'wb')
#content = originalf.read()
#outsize = 0
#for line in content.splitlines():
#    outsize += len(line) + 1
#    modifiedf.write(line + str.encode('\n'))

#print('wrote %s bytes' % (len(content) - outsize))
#originalf.close()
#modifiedf.close()
### Load the dictionary containing the dataset
data_file =  open(modified, 'rb')
data_dict = pickle.load(data_file)

### Task 2: Remove outliers
for key, value in data_dict.items():
    print("key: ", key,  ", value: ", value)
    print('---------------------------a-')
print('total rows: ' + str(len(data_dict.keys())))

data = pd.DataFrame.from_dict(data_dict, orient = 'index')
print("data frame")
print(data.head())

meand = data.mean()
print(meand.head())