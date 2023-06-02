import numpy as np
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logfile', type=str,
                    help="Log file name")
args = parser.parse_args()
file_name = args.logfile
f = open(file_name+'.txt')
st = f.read()
st = st.split('\n')

dataset = re.compile(r'dataset:(.*?)\'')
DATASET = dataset.findall(str(st))

patch = re.compile(r'patch_size:(.*?)\'')
PATCH = patch.findall(str(st))

finetune = re.compile(r'fine_tune:(.*?)\'')
FINETUNE = finetune.findall(str(st))

desc = re.compile(r'desc:(.*?)\'')
DESC = desc.findall(str(st))

percentage = re.compile(r'load_data:(.*?)\'')
PER = percentage.findall(str(st))

kappa = re.compile(r'Kappa:\', \'(.*?)\',')
KAPPA = kappa.findall(str(st))

accuracy = re.compile(r'Accuracy:\', \'(.*?)\',')
ACCURACY = accuracy.findall(str(st))

aa = re.compile(r'F1 scores:\',\s\'\[(.*?)\]')
AA = aa.findall(str(st))

AA_list = []
F1 = []
for i in range(len(AA)):
    lst = AA[i].strip().split('\',\s\'')[0].replace('\'', '').replace(' ', ',').replace(',', ' ').split(' ')
    while '' in lst:
        lst.remove('')
    lst = np.array(lst, np.float32)
    F1.append(lst[1:])
    AA_list.append(sum(lst[1:])/(lst.shape[0]-1))

len_DATASET = len(DATASET)
DATASET = np.array(DATASET).reshape([len_DATASET,1])

len_PATCH = len(PATCH)
PATCH = np.array(PATCH).reshape([len_PATCH,1])

len_FINETUNE = len(FINETUNE)
FINETUNE = np.array(FINETUNE).reshape([len_FINETUNE,1])

len_DESC = len(DESC)
DESC = np.array(DESC).reshape([len_DESC,1])

len_PER = len(PER)
PER = np.array(PER).reshape([len_PER,1])

len_OA = len(ACCURACY)
OA = np.array(ACCURACY).reshape([len_OA,1])

len_AA = len(AA)
AA = np.array(AA_list).reshape([len_AA,1])

len_KAPPA = len(KAPPA)
KAPPA = np.array(KAPPA).reshape([len_KAPPA,1])

len_F1 = len(F1)
F1 = np.array(F1).reshape([len_AA, len(F1[0])])

len_list = [len_DATASET, len_PER, len_OA, len_AA, len_KAPPA]
print(len_list)
if np.var(len_list)!=0.0:
    raise ValueError("{}Inconsistent data length.".format(len_list))
#DATA = np.hstack((DATASET, PATCH, PER, OA, AA, KAPPA, F1))
DATA = np.hstack((DATASET, PATCH, FINETUNE, DESC, PER, OA, AA, KAPPA, F1))

df = pd.DataFrame(DATA)
# IndianPines
# df.columns = ['DATASET',  'PATCH', 'FINETUNE', 'DESC', 'PER', 'OA', 'AA', 'KAPPA', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
# Botswana
df.columns = ['DATASET',  'PATCH', 'FINETUNE', 'DESC', 'PER', 'OA', 'AA', 'KAPPA', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    
df.to_excel(file_name+'.xlsx', sheet_name='Sheet1', index=False)

