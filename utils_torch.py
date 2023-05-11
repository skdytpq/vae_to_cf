import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import json
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pdb
import random
DEFAULT_SEED = 42
SEED = DEFAULT_SEED
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# Auxiliary functions

def set_seed(seed=DEFAULT_SEED):
    """
    Set random seed in all used libs.
    """
    global SEED
    SEED = seed
    np.random.seed(seed)
    random.seed(seed)


def get_seed():
    return SEED


def shufflestr(x):
    """
    Shuffle randomly items in string separated by comma.
    """
    p = x.split(',') # , 로 구분된 아이템을 섞은 후 다시 return
    random.shuffle(p)
    return ",".join(p)


def split2_50(x):
    """
    Returns first half of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .5)
    return ",".join(p[:s])


def split1_50(x):
    """
    Returns second half of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .5)
    return ",".join(p[s:])


def split75(x):
    """
    Returns first three quarters of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .75)
    return ",".join(p[:s])


def split25(x):
    """
    Returns last quarter of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .75)
    return ",".join(p[s:])


def getDataloader(dataset, train_file, val_file, test_file, item_sorted,batch_size):
    with open(item_sorted) as f:
        item = json.load(f)
    item_sorted = pd.DataFrame(item)
    toki = CountVectorizer() # 텍스트 토크나이저 불러오기
    arrg = toki.fit_transform((item_sorted.itemid))
    _, num_words = arrg.toarray().shape
    print(num_words , ': Number of words')

    if dataset == 'movielens':
        train_loader = torch.utils.data.DataLoader(
                                            Movielens(train_file,'train',toki, batch_size  = batch_size),
                                            batch_size = 1 ,shuffle=False,)

        val_loader   = torch.utils.data.DataLoader(
                                            Movielens(val_file,'val',toki, batch_size  = batch_size),
                                             batch_size = 1 ,shuffle=False,)

        test_loader =  torch.utils.data.DataLoader(
                                            Movielens(test_file,'test',toki, batch_size  = batch_size),
                                         batch_size = 1 ,  shuffle=False,)
    else:
        train_loader,val_loader,test_loader = None,None,None


    return train_loader, val_loader, test_loader

class Movielens(data.Dataset):
    def __init__(self, txt_file,mode,toki,batch_size):
        self.mode = mode
        self.full_data=False
        self.p50_splits=True
        self.p2575_splits=False
        self.p7525_splits=False
        self.p2525_splits=False
        self.p7575_splits=False
        self.prevent_identity=True
        self.txt_file = pd.read_csv(txt_file)
        self.data_np = self.txt_file
        self.toki = toki
        self.batch_size = batch_size
        if self.mode != 'train':
            self.batch_size = 128
        self.random_batching = False
        if self.mode == 'train':
            self.randombatching = True

    def __len__(self):
        return int(np.floor(self.data_np.shape[0] / self.batch_size)) - 1
        
    def __getitem__(self, index):
        # binary mode = output vectors is 0/1 only
        mod = 'binary'
        if index == 0:
            if self.random_batching == True:
                self.data_np = self.data_np.sample(frac=1)
            self.data_np['temp_itemids_p'] = self.data_np['itemids'].apply(shufflestr) # 아이템 list 를 섞은 후 다시 return 
            self.data_np['temp_itemids_p1_50'] = self.data_np['temp_itemids_p'].apply(split1_50) # 각 비율로 자르기 
            self.data_np['temp_itemids_p2_50'] = self.data_np['temp_itemids_p'].apply(split2_50)
            self.data_np['temp_itemids_p_25'] = self.data_np['temp_itemids_p'].apply(split25)
            self.data_np['temp_itemids_p_75'] = self.data_np['temp_itemids_p'].apply(split75)
            self.data_np = self.data_np.to_numpy()
        try:
            data_slice = self.data_np[self.batch_size * index:self.batch_size * index + self.batch_size] # 1024
        except:
            data_slice = self.data_np[self.batch_size * index : -1]
        # 한 배치 불러오기  
        indices = list(range(self.__len__())) # 전체 길이 / 배치사이즈 => 총 나눠지는 인덱스의 수 
        indices += indices
        # 2*indices
        index2 = indices[index + 1] # 각 다음 배치 할당
        index3 = indices[index + 2]
        index4 = indices[index + 3]
        index5 = indices[index + 4]
        if self.mode != 'train':
            self.prevent_identity = False
            self.full_data = True
            self.p50_splits = False
        
        if self.full_data:
            data_slice = self.data_np[self.batch_size * index:self.batch_size * index + self.batch_size]

        if self.p50_splits:
            data_slice2 = self.data_np[self.batch_size * index2:self.batch_size * index2 + self.batch_size]
            data_slice3 = self.data_np[self.batch_size * index3:self.batch_size * index3 + self.batch_size]

        if self.p2575_splits or self.p7525_splits or self.p2525_splits or self.p7575_splits:
            data_slice4 = self.data_np[self.batch_size * index4:self.batch_size * index4 + self.batch_size]
            data_slice5 = self.data_np[self.batch_size * index5:self.batch_size * index5 + self.batch_size]
        ret_x = []
        ret_y = []

        # full input to full_output
        if self.full_data:
            ret_x.append(self.toki.transform(data_slice[:, 1]).toarray()) # 각 data 의 item token화
            ret_y.append(self.toki.transform(data_slice[:, 1]).toarray())

        if self.p50_splits:
            ret_x.append(self.toki.transform(data_slice2[:, 3]).toarray())
            ret_x.append(self.toki.transform(data_slice3[:, 4]).toarray())

            if self.prevent_identity:
                ret_y.append(self.toki.transform(data_slice2[:, 4]).toarray())
                ret_y.append(self.toki.transform(data_slice3[:, 3]).toarray())
            else:
                ret_y.append(self.toki.transform(data_slice2[:, 3]).toarray())
                ret_y.append(self.toki.transform(data_slice3[:, 4]).toarray())

        if self.p2575_splits:
            ret_x.append(self.toki.transform(data_slice4[:, 5]).toarray())
            ret_y.append(self.toki.transform(data_slice4[:, 6]).toarray())

        if self.p7525_splits:
            ret_x.append(self.toki.transform(data_slice4[:, 6]).toarray())
            ret_y.append(self.toki.transform(data_slice4[:, 5]).toarray())

        if self.p2525_splits:
            ret_x.append(self.toki.transform(data_slice5[:, 5]).toarray())
            ret_y.append(self.toki.transform(data_slice5[:, 5]).toarray())

        if self.p7575_splits:
            ret_x.append(self.toki.transform(data_slice5[:, 6]).toarray())
            ret_y.append(self.toki.transform(data_slice5[:, 6]).toarray())
        return np.vstack(ret_x), np.vstack(ret_y)
