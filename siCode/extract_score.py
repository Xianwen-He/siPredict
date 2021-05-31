import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import pandas as pd
import warnings
from utils.siScore_utils import *
from utils.parameters import *
import os
import numpy as np
import time
# torch.cuda.empty_cache()  # release memory
warnings.filterwarnings("ignore")

args = extract_score_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(nn.Linear(512, 1))

### load trained model
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) 
model_path = os.path.join('./checkpoint', args.model)
model.load_state_dict(torch.load(model_path)['model'])
model.to(device)
print("Load Finished")

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        # 所有图片所在路径
        # 默认路径已修改
        self.file_list = glob.glob(test_dir+'/*.png')
        self.transform = transform        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        # 获取图片y-x名称
        # 图片尾缀已修改
        name = path.split("/")[-1].split("-2017.png")[0]
        # 转化为3通道RGB图像
        image = np.delete(io.imread(path), [3], axis = 2 )/ 255.0
        if self.transform:
            image = self.transform(np.stack([image])).squeeze()
        return image, name

# To enforce the batch normalization during the evaluation
model.eval()    
    
### Testing part
_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]
test_dataset = TestDataset(test_dir = args.test_dir,
                           transform = transforms.Compose([ToTensor(), Normalize(mean=_mean, std=_std)]))
# num_workers = 0
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_sz, shuffle=False, num_workers=0)
print('number of test images:', len(test_dataset))

# 已修改路径与算法，仅关注score
# grid_df: columns: [y-x, ...]
grid_df = pd.read_csv(args.test)
grid_df['predict'] = -1  # 图片得分
# print(grid_df.loc[0])
print('grid dataset loaded')

with torch.no_grad():
    for batch_idx, (data, name) in enumerate(test_loader):
        if batch_idx%10 == 0:
            print('batch index:', batch_idx)
        # print(name)
        data = data.to(device)
        scores = model(data).squeeze()
        # print('score extracted!', type(scores), scores)
        # time.sleep(10)
        count = 0
        for each_name in name:
            # print('current score:', type(scores[count].cpu().data.numpy()), scores[count].cpu().data.numpy())
            # time.sleep(5)
            # 已修改列名
            grid_df.loc[grid_df['y-x'] == each_name, 'predict'] = scores[count].cpu().data.numpy().tolist()
            count += 1

# 存储结果
# 已经修改args参数与此处操作
df_predicted = grid_df.loc[grid_df['predict'] != -1]  
df_predicted.to_csv(args.predict_scores, index = False, header = True) 
