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
    
    for i,a in tqdm(enumerate(prices)):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = df[[a,b,c]].max(axis=1);
                    min_ = df[[a,b,c]].min(axis=1);
                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_;

                    df[f'{a}_{b}_{c}_imb2'] = ((max_-mid_)/(mid_-min_)).astype(np.float32);
                    features.append(f'{a}_{b}_{c}_imb2');
    
    return df[features];

# 根据交易量计算权重
def create_weights(df: pd.DataFrame, is_test=False)->pd.DataFrame:
    def get_stock_weight(data_batch):
        sizes=data_batch['matched_size']
        waps=data_batch['wap']
        matched_volume=sizes*waps
        total_vol=matched_volume.sum()
        weights=matched_volume/total_vol
        out=data_batch.copy()
        out['weights']=weights
        #print(out)
        return out
    
    df=df.copy()
    
    if is_test:
        return get_stock_weight(df)
    else:
        for time in tqdm(df['time_id'].unique()):
            input_df=df.query(f"time_id=={time}")
            out_weigth_df=get_stock_weight(input_df)
            df.loc[out_weigth_df.index,"weights"]=out_weigth_df['weights']
        print(f"Last batch weigth: {out_weigth_df['weights']}")
        print(f"Sum of the weigths: {out_weigth_df['weights'].sum()}")
        print("Ending weigth assign...")
        return df

"""Credits to https://www.kaggle.com/code/lblhandsome/optiver-robust-best-single-model#Feature-groups"""


@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            if mid_val == min_val:  # Prevent division by zero
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

# generate imbalance features ##Robust functions
def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features

def imbalance_features(df):
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    # V1
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
        
    # V2
    df["stock_weights"] = df["stock_id"].map(weights)
    df["weighted_wap"] = df["stock_weights"] * df["wap"]
    df['wap_momentum'] = df.groupby('stock_id')['weighted_wap'].pct_change(periods=6)
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    df['spread_depth_ratio'] = (df['ask_price'] - df['bid_price']) / (df['bid_size'] + df['ask_size'])
    df['mid_price_movement'] = df['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['micro_price'] = ((df['bid_price'] * df['ask_size']) + (df['ask_price'] * df['bid_size'])) / (df['bid_size'] + df['ask_size'])
    df['relative_spread'] = (df['ask_price'] - df['bid_price']) / df['wap']
    
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    # V3
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
            
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size',
                'wap', 'near_price', 'far_price']:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    return df.replace([np.inf, -np.inf], 0)

# generate time & stock features
def other_features(df,is_train=True,global_path=None):
    df["dow"] = df["date_id"] % 5
    df["dom"] = df["date_id"] % 20
    df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60

    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

def generate_all_features(df,is_train=True,global_path=None):
    cols = [c for c in df.columns if c not in ["row_id", "time_id",'target']]
    df = df[cols]
    df = imbalance_features(df)
    df = other_features(df,is_train,global_path)
    gc.collect()
    
    feature_name = [i for i in df.columns if i not in ["row_id", "time_id", "date_id"]]
    
    return df[feature_name]

# weights的含义是什么？
weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]

weights = {int(k):v for k,v in enumerate(weights)}

# print("Loaded preprocess funtions");

# Loaded preprocess funtions
class Manifold_prep:
    #Manifold imbalance features
    @classmethod
    def imbalance_features(cls,df):
        # Define lists of price and size-related column names
        print('Calling manifold imbalance function')
        prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
        sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
        df["volume"] = df.eval("ask_size + bid_size")
        df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
        df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
        df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
        df["size_imbalance"] = df.eval("bid_size / ask_size")

        for c in combinations(prices, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

        for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
            triplet_feature = calculate_triplet_imbalance_numba(c, df)
            df[triplet_feature.columns] = triplet_feature.values

        df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
        df["price_spread"] = df["ask_price"] - df["bid_price"]
        df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
        df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
        df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
        df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])

        # Calculate various statistical aggregation features
        for func in ["mean", "std", "skew", "kurt"]:
            df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
            df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)


        for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
            for window in [1, 2, 3, 10]:
                df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
                df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)

        # Calculate diff features for specific columns
        for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum', 'size_imbalance']:
            for window in [1, 2, 3, 10]:
                df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

        return df.replace([np.inf, -np.inf], 0)
    
    @classmethod
    def other_features(cls,df,is_train=True,global_path=None):
        df["dow"] = df["date_id"] % 5  # Day of the week
        df["seconds"] = df["seconds_in_bucket"] % 60  
        df["minute"] = df["seconds_in_bucket"] // 60  
        for key, value in global_stock_id_feats.items():
            df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

        return df
    
    @classmethod
    def generate_all_features(cls,df,is_train=True,global_path=None):
        # Select relevant columns for feature generation
        cols = [c for c in df.columns if c not in ["row_id", "time_id","target","currently_scored"]]
        df = df[cols]

        # Generate imbalance features
        df = cls.imbalance_features(df)
        df = cls.other_features(df)
        gc.collect()
        feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]

        return df[feature_name]

# Eval prep
def eval_prep(expresion:str,df):
    global dfs_evaluated
    df=df.copy()
    args=expresion.split('|')
    prep_func=args[0]
    
    if len(dfs_evaluated.keys())!=0:
        if list(dfs_evaluated.keys())[0] != prep_func.split('~')[0]:
            dfs_evaluated={}
        
    for func in prep_func.split('~'):
        if func=='feat_eng':
            if 'feat_eng' in dfs_evaluated.keys():
                df=dfs_evaluated['feat_eng']
            else:
                df=feat_eng(df,is_train=False,feats_path='/kaggle/input/optiver-train-preprocessed')
                dfs_evaluated['feat_eng']=df
        elif func=='ftree':
            if 'ftree' in dfs_evaluated.keys():
                df=dfs_evaluated['ftree']
            else:
                df=MakeFtre(df)
                dfs_evaluated['ftree']=df
        elif func=='weights':
            if 'weights' in dfs_evaluated.keys():
                df=dfs_evaluated['weights']
            else:
                df=create_weights(df,True)
                dfs_evaluated['weights']=df
        elif func=='robust':
            if 'robust' in dfs_evaluated.keys():
                df=dfs_evaluated['robust']
            else:
                df=generate_all_features(df,is_train=False,global_path='/kaggle/input/optiver-train-preprocessed/global_stock_id_feats.dict')
                dfs_evaluated['robust']=df
                
        elif func=='manifold':
            if 'manifold' in dfs_evaluated.keys():
                df=dfs_evaluated['manifold']
            else:
                df=Manifold_prep.generate_all_features(df,is_train=False,global_path='/kaggle/input/optiver-train-preprocessed/global_stock_id_feats.dict')
                dfs_evaluated['manifold']=df
    return df

# Ensemble fold functions
def cube(x):
    if x >= 0:
        return x**(1/3)
    elif x < 0:
        return -(abs(x)**(1/3))
    
def pow_order(x,n):
    if x >= 0:
        return x**(n)
    elif x < 0:
        return -(abs(x)**(n))
    

def cubic_mean(array_results,axis=0):
    array_results=array_results.copy()
    columns=[i for i in range(len(array_results))]
    df_results=pd.DataFrame(np.vstack(array_results).T)
    
    df_results.columns=columns.copy()
    cubic_cols=[]
    for col in columns:
        df_results[f'{col}_3']=df_results[col]**3
        cubic_cols.append(f'{col}_3')
        
    #if CFG.name=='manifold|manifold|lgbm':
        #print(f'\n the cubic fold dataframe results are: \n{df_results}')
    return df_results[cubic_cols].mean(axis=1).apply(cube).values

def pow_mean(array_results,order=0):
    array_results=array_results.copy()
    columns=[i for i in range(len(array_results))]
    df_results=pd.DataFrame(np.vstack(array_results).T)
    
    df_results.columns=columns.copy()
    cubic_cols=[]
    for col in columns:
        df_results[f'{col}_{order}']=df_results[col].apply(lambda x:pow_order(x,order))
        cubic_cols.append(f'{col}_{order}')
        
    return df_results[cubic_cols].mean(axis=1).apply(lambda x:pow_order(x,1/order)).values


def load_models(models_dir,n_splits,cv_split,alias_prep):
    models=[]
    for fold in range(n_splits):
        model=lgb.Booster(model_file=f'{models_dir}/fold_{fold}_{cv_split}{alias_prep}.model')
        models.append(model)
    return models

class CFGg:
    # TODO: update the path
    # base_train=pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
    base_train=pd.read_csv('../train.csv')
    target_val=base_train['target'].dropna().reset_index(drop=True)
    is_train=False
    ensemble_function=pow_mean
    pow_order=0.5
    
#This is the actual best model
class CFG6:
    models_dir='/kaggle/input/v6-optiver-trained/lgbm_models'
    oof_path='/kaggle/input/v5-optiver-oof-df/oof_df_manifold.pkl'
    cv_split='manifold'
    n_splits=11
    weight=1.08
    name='manifold|manifold|lgbm'
    drop_columns=[]
    alias_prep='_robust'
    models=load_models(models_dir,n_splits,cv_split,alias_prep)

    
CFGlist=[CFG6]

if not CFGg.is_train:
    global_stock_id_feats=joblib.load('/kaggle/input/optiver-train-preprocessed/global_stock_id_feats.dict')


# Inference
def predict_fold(models,test_data,drop_columns):
    preds=[]
    for model in models:
        #DEBUG_DF_COLS(test_data)
        
        pred_fold=model.predict(test_data.drop(columns=drop_columns))
        preds.append(pred_fold)
    if CFG.name=='manifold|manifold|lgbm':
        print(f'The manifold fold preds are: \n{pd.DataFrame(np.vstack(preds).T)}')
    prediction=CFGg.ensemble_function(preds,CFGg.pow_order)
    return prediction

import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()

import warnings
import time
warnings.filterwarnings("ignore")

def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices) / np.sum(std_error)
    out = prices - std_error * step
    return out

y_min, y_max = -64, 64

cache = pd.DataFrame()
counter=0
qps=[]
dfs_evaluated={}

for (test, revealed_targets, sample_prediction) in iter_test:
    now_time = time.time()
    preds={}
    cache = pd.concat([cache, test], ignore_index=True, axis=0)
    
    if counter > 0:
        cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
    
    for CFG in CFGlist:
        #print(CFG.name,dfs_evaluated.keys())
        #scored_index=
        st=time.time()
        prep_test=eval_prep(CFG.name,cache)[-len(test):]
        print(f'preprocess time: {time.time()-st}')
        st=time.time()
        preds[CFG.name]=predict_fold(CFG.models,prep_test,CFG.drop_columns)*CFG.weight
        print(f'{CFG.name} inference time: {time.time()-st}')

    #Make a dataframe of the results
    df_results=pd.DataFrame(preds)
    
    #Calculates the sum of the weighted results
    target_result=df_results.sum(axis=1)
    
    lgb_predictions = zero_sum(target_result.values, test['bid_size'] + test['ask_size'])
    clipped_predictions = np.clip(lgb_predictions, y_min, y_max)
    
    #Makes and debug the targets and results
    print(revealed_targets.head(10)['revealed_target'])
    #print(revealed_targets)
    
    sample_prediction['target'] = clipped_predictions
    env.predict(sample_prediction)
    
    #Important to save runtime excecution
    dfs_evaluated={}
    counter += 1
    
    qps.append(time.time() - now_time)
    if counter % 10 == 0:
        print(counter, 'qps:', np.mean(qps))
            
time_cost = 1.146 * np.mean(qps)
print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")

if __name__ == '__main__':
    is_offline = False

    if is_offline:
    
        df_train = df[df["date_id"] <= split_day]
        df_valid = df[df["date_id"] > split_day]
        print("Offline mode")
        print(f"train : {df_train.shape}, valid : {df_valid.shape}")
    else:
        df_train = df
        print("Online mode")
        if is_train:
            global_stock_id_feats = {
                "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
                "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
                "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
                "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
                "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
                "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
            }
            if is_offline:
                df_train_feats = generate_all_features(df_train)
                print("Build Train Feats Finished.")
                df_valid_feats = generate_all_features(df_valid)
                print("Build Valid Feats Finished.")
                df_valid_feats = reduce_mem_usage(df_valid_feats)
            else:
                df_train_feats = generate_all_features(df_train)
                print("Build Online Train Feats Finished.")

            df_train_feats = reduce_mem_usage(df_train_feats)

    lgb_params = {
        "objective": "mae",
        "n_estimators": 6000,
        "num_leaves": 256,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "learning_rate": 0.00871,
        'max_depth': 11,
        "n_jobs": 4,
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
    }
    feature_name = list(df_train_feats.columns)
    print(f"Feature length = {len(feature_name)}")

    num_folds = 5
    fold_size = 480 // num_folds
    gap = 5

    models = []
    scores = []

    model_save_path = 'modelitos_para_despues' 
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    date_ids = df_train['date_id'].values

    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size
        if i < num_folds - 1:  # No need to purge after the last fold
            purged_start = end - 2
            purged_end = end + gap + 2
            train_indices = (date_ids >= start) & (date_ids < purged_start) | (date_ids > purged_end)
        else:
            train_indices = (date_ids >= start) & (date_ids < end)
        
        test_indices = (date_ids >= end) & (date_ids < end + fold_size)
        
        df_fold_train = df_train_feats[train_indices]
        df_fold_train_target = df_train['target'][train_indices]
        df_fold_valid = df_train_feats[test_indices]
        df_fold_valid_target = df_train['target'][test_indices]

        print(f"Fold {i+1} Model Training")
        
        # Train a LightGBM model for the current fold
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            df_fold_train[feature_name],
            df_fold_train_target,
            eval_set=[(df_fold_valid[feature_name], df_fold_valid_target)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=100),
            ],
        )

        models.append(lgb_model)
        # Save the model to a file
        model_filename = os.path.join(model_save_path, f'doblez_{i+1}.txt')
        lgb_model.booster_.save_model(model_filename)
        print(f"Model for fold {i+1} saved to {model_filename}")

        # Evaluate model performance on the validation set
        fold_predictions = lgb_model.predict(df_fold_valid[feature_name])
        fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
        scores.append(fold_score)
        print(f"Fold {i+1} MAE: {fold_score}")

        # Free up memory by deleting fold specific variables
        del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
        gc.collect()

    # Calculate the average best iteration from all regular folds
    average_best_iteration = int(np.mean([model.best_iteration_ for model in models]))

    # Update the lgb_params with the average best iteration
    final_model_params = lgb_params.copy()
    final_model_params['n_estimators'] = average_best_iteration

    print(f"Training final model with average best iteration: {average_best_iteration}")

    # Train the final model on the entire dataset
    final_model = lgb.LGBMRegressor(**final_model_params)
    final_model.fit(
        df_train_feats[feature_name],
        df_train['target'],
        callbacks=[
            lgb.callback.log_evaluation(period=100),
        ],
    )

    # Append the final model to the list of models
    models.append(final_model)

    # Save the final model to a file
    final_model_filename = os.path.join(model_save_path, 'doblez-conjunto.txt')
    final_model.booster_.save_model(final_model_filename)
    print(f"Final model saved to {final_model_filename}")

    # Now 'models' holds the trained models for each fold and 'scores' holds the validation scores
    print(f"Average MAE across all folds: {np.mean(scores)}")

    def zero_sum(prices, volumes):
        std_error = np.sqrt(volumes)
        step = np.sum(prices) / np.sum(std_error)
        out = prices - std_error * step
        return out

    if is_infer:
        import optiver2023
        env = optiver2023.make_env()
        iter_test = env.iter_test()
        counter = 0
        y_min, y_max = -64, 64
        qps, predictions = [], []
        cache = pd.DataFrame()

        # Weights for each fold model
        model_weights = [1/len(models)] * len(models) 
        
        for (test, revealed_targets, sample_prediction) in iter_test:
            now_time = time.time()
            cache = pd.concat([cache, test], ignore_index=True, axis=0)
            if counter > 0:
                cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
            feat = generate_all_features(cache)[-len(test):]

            # Generate predictions for each model and calculate the weighted average
            lgb_predictions = np.zeros(len(test))
            for model, weight in zip(models, model_weights):
                lgb_predictions += weight * model.predict(feat)

            lgb_predictions = zero_sum(lgb_predictions, test['bid_size'] + test['ask_size'])
            clipped_predictions = np.clip(lgb_predictions, y_min, y_max)
            sample_prediction['target'] = clipped_predictions
            env.predict(sample_prediction)
            counter += 1
            qps.append(time.time() - now_time)
            if counter % 10 == 0:
                print(counter, 'qps:', np.mean(qps))

        time_cost = 1.146 * np.mean(qps)
        print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")