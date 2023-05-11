from utils import *
import tensorflow as tf
import tensorflow_addons as tfa
import pdb

set_seed(42)

dataset = Data(d='../../data/', pruning='u5')
dataset.splits = []
dataset.create_splits(1, 10000, shuffle=False, generators=False)
dataset.split.train_users = pd.read_json("../../data/train_users.json").userid.apply(str).to_frame()
dataset.split.validation_users = pd.read_json("../../data/val_users.json").userid.apply(str).to_frame()
dataset.split.test_users = pd.read_json("../../data/test_users.json").userid.apply(str).to_frame()
dataset.split.generators()
a = dataset.split.generators()
pdb.set_trace()
