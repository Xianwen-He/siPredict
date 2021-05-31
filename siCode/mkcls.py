import os
import shutil
import pandas as pd
import numpy as np
import argparse


suffix = '-2017.png'
Data_dir = '../Data'

def mkcls_parser():
    parser = argparse.ArgumentParser(description='mkcls parser')
    parser.add_argument('--img_dir', default='/home/liuhaoyu/DATA2/siImages/2017', type=str, help='original image path')
    parser.add_argument('--cluster_dir', default='cluster', type=str, help='cluster path')
    parser.add_argument('--cluster_unified_dir', default = 'cluster_unified', type=str, help='unified cluster path')
    parser.add_argument('--meta_data', default = 'grid.csv', type=str, help='grid data path')
    
    return parser.parse_args()

def main(args):
    meta_df = pd.read_csv(os.path.join(Data_dir, args.cluster_dir, args.meta_data))
    print('meta data loaded,', meta_df.shape[0], 'images')
    
    cluster_ids = np.unique(meta_df['cluster_id'])
    # 生成文件夹
    for cid in cluster_ids:
        os.mkdir(os.path.join(Data_dir, args.cluster_dir, str(cid)))
    print('cluster paths generated')
    
    # 复制图片
    for i in range(meta_df.shape[0]):
        img = meta_df['y-x'][i] + suffix
        cid = meta_df['cluster_id'][i]
        
        shutil.copyfile(os.path.join(args.img_dir, img), os.path.join(Data_dir, args.cluster_dir, str(cid), img))
        shutil.copyfile(os.path.join(args.img_dir, img), os.path.join(Data_dir, args.cluster_unified_dir, img))
        
        if (i+1)%100 == 0:
            print(i+1, 'images finished')
            
    print('all finished')

if __name__ == '__main__':
    args = mkcls_parser()
    main(args)
