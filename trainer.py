import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
from sklearn.model_selection import train_test_split

import glob

dir_path = "data/train"
filename = glob.glob(dir_path + "/*.csv")

df = pd.read_csv(filename[0])
print(df)

# # 컬럼명 영어로 변경
# df.columns = ["stn_no","stn_name","date","temperature","land_temperature"]
# # 필요없는 컬럼 삭제
# df = df.drop(columns=["stn_no", "stn_name", "date"])