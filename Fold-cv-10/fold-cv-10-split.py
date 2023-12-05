import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from itertools import combinations
import gc
from sklearn.linear_model import LinearRegression, Lasso
import lightgbm as lgb
import warnings
from numba import njit, prange
warnings.filterwarnings("ignore")

from tqdm import tqdm
import sys
from numba import njit, prange

def feat_eng(df,is_train=True,feats_path=None):
    if is_train:
        median_sizes = df.groupby('stock_id')['bid_size'].median() + df.groupby('stock_id')['ask_size'].median()
        std_sizes = df.groupby('stock_id')['bid_size'].std() + df.groupby('stock_id')['ask_size'].std()
        min_sizes = df.groupby('stock_id')['bid_size'].min() + df.groupby('stock_id')['ask_size'].min()
        max_sizes = df.groupby('stock_id')['bid_size'].max() + df.groupby('stock_id')['ask_size'].max()
        joblib.dump(median_sizes,'median.feat')
        joblib.dump(std_sizes,'std.feat')
        joblib.dump(min_sizes,'min.feat')
        joblib.dump(max_sizes,'max.feat')
    else:
        median_sizes = joblib.load(f'{feats_path}/median.feat')
        std_sizes = joblib.load(f'{feats_path}/std.feat')
        min_sizes = joblib.load(f'{feats_path}/min.feat')
        max_sizes = joblib.load(f'{feats_path}/max.feat')

    cols = [c for c in df.columns if c not in ['row_id','time_id']]
    df = df[cols]
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    df['bid_ask_volume_diff'] = df['ask_size'] - df['bid_size']
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
    df['bid_plus_ask_sizes'] = df['bid_size'] + df['ask_size']
    df['median_size'] = df['stock_id'].map(median_sizes.copy().to_dict())
    df['std_size'] = df['stock_id'].map(std_sizes.copy().to_dict())
    df['min_size'] = df['stock_id'].map(median_sizes.copy().to_dict())
    df['max_size'] = df['stock_id'].map(std_sizes.copy().to_dict())
    df['high_volume'] = np.where(df['bid_plus_ask_sizes'] > df['median_size'], 1, 0)
        
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    print("Making combinations of the prices")
    for c in tqdm(combinations(prices, 2)):
        df[f'{c[0]}_minus_{c[1]}'] = (df[f'{c[0]}'] - df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_{c[1]}_imb'] = df.eval(f'({c[0]}-{c[1]})/({c[0]}+{c[1]})')
    
    print("Making combinations of the prices")
    for c in tqdm(combinations(prices, 3)):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1)-min_-max_

        df[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
        
    gc.collect()
    
    return df

def MakeFtre(df : pd.DataFrame) -> pd.DataFrame:
    features = [
                'stock_id','date_id','seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ];
    prices = ['reference_price', 'far_price', 'near_price', 'bid_price', 'ask_price', 'wap']
    
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32);
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32);
       
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})');
                features.append(f'{a}_{b}_imb'); 

