import gc
import os
import time
import sys
import warnings
from itertools import combinations
from warnings import simplefilter
import joblib
import lightgbm as lgb
import numpy as np
import json
import pandas as pd
import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
import polars as pl
import logging
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



is_offline = False
LGB = True
NN = False
is_train = True
is_infer = True
max_lookback = np.nan
split_day = 435
base_dir = '/home/joseph/Projects/Optiver---Trading-at-the-close'
model_name = 'Feats161'
log_dir = 'logs'
results_dir = 'results'



exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
graph_name = "%s_split%d_time%s" % \
                 (model_name, split_day, exp_time)



# Utilities
def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2**(n + 1 - j)))
    return w

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

if not os.path.exists(os.path.join(base_dir, log_dir)):
    os.mkdir(os.path.join(base_dir, log_dir))

logger = get_logger(os.path.join(base_dir, log_dir), "test", log_filename=graph_name + ".log")

logger.info(f'is_offline {is_offline}')
logger.info(f'LGB {LGB}')
logger.info(f'NN {NN}')
logger.info(f'model_name {model_name}')
logger.info(f'is_train {is_train}')
logger.info(f'is_infer {is_infer}')
logger.info(f'max_lookback {max_lookback}')
logger.info(f'split_day {split_day}')



from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]
    
def reduce_mem_usage(df, verbose=0):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
               
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")
    return df

# DataLoading
df = pd.read_csv(os.path.join(base_dir, "train.csv"))
#os.path.join(base_dir, log_dir)
df = df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df_shape = df.shape

logger.info(f"df_shape: {df_shape}")

### Parallel Triplet Imbalance Calculation

from numba import njit, prange

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
            
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features

### Feature Generation Functions

def imbalance_features(df):
    # Define lists of price and size-related column names
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
    
    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1,3,5,10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'weighted_wap','price_spread']:
        for window in [1,3,5,10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    
    #V4 feature
    for window in [3,5,10]:
        df[f'price_change_diff_{window}'] = df[f'bid_price_diff_{window}'] - df[f'ask_price_diff_{window}']
        df[f'size_change_diff_{window}'] = df[f'bid_size_diff_{window}'] - df[f'ask_size_diff_{window}']

    #V5 - rolling diff
    # Convert from pandas to Polars
    pl_df = pl.from_pandas(df)

    #Define the windows and columns for which you want to calculate the rolling statistics
    windows = [3, 5, 10]
    columns = ['ask_price', 'bid_price', 'ask_size', 'bid_size']

    # prepare the operations for each column and window
    group = ["stock_id"]
    expressions = []

    # Loop over each window and column to create the rolling mean and std expressions
    for window in windows:
        for col in columns:
            rolling_mean_expr = (
                pl.col(f"{col}_diff_{window}")
                .rolling_mean(window)
                .over(group)
                .alias(f'rolling_diff_{col}_{window}')
            )

            rolling_std_expr = (
                pl.col(f"{col}_diff_{window}")
                .rolling_std(window)
                .over(group)
                .alias(f'rolling_std_diff_{col}_{window}')
            )

            expressions.append(rolling_mean_expr)
            expressions.append(rolling_std_expr)

    # Run the operations using Polars' lazy API
    lazy_df = pl_df.lazy().with_columns(expressions)

    # Execute the lazy expressions and overwrite the pl_df variable
    pl_df = lazy_df.collect()

    # Convert back to pandas if necessary
    df = pl_df.to_pandas()
    gc.collect()
    
    df['mid_price*volume'] = df['mid_price_movement'] * df['volume']
    df['harmonic_imbalance'] = df.eval('2 / ((1 / bid_size) + (1 / ask_size))')
    
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    return df

def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    df['time_to_market_close'] = 540 - df['seconds_in_bucket']

    # new feature
    df['before_5min'] = 1
    df.loc[df['seconds_in_bucket']>=300,'before_5min'] = -1
    
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    # Generate imbalance features
    df = imbalance_features(df)
    gc.collect() 
    df = other_features(df)
    gc.collect()  
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    logger.info(f"feats_length:{len(feature_name)}")
    
    return df[feature_name]

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

### Data Splitting

if is_offline:
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    logger.info("Offline mode")
    logger.info(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    df_train = df
    logger.info("Online mode")

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
        logger.info("Build Train Feats Finished.")
        df_valid_feats = generate_all_features(df_valid)
        logger.info("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)
    else:
        df_train_feats = generate_all_features(df_train)
        logger.info("Build Online Train Feats Finished.")

    df_train_feats = reduce_mem_usage(df_train_feats)

### Model Training

### LGB

if LGB:
    import numpy as np
    import lightgbm as lgb
    
    lgb_params = {
        "objective": "mae",
        "n_estimators": 10000,
        "num_leaves": 256,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        #"learning_rate": 0.00971,
        "learning_rate": 0.01,
        'max_depth': 11,
        "n_jobs": 32,
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
        #"reg_alpha": 0.1,
        "reg_alpha": 0.2,
        "reg_lambda": 3.25
    }
    logger.info(f"lgb_params: {lgb_params}")

    feature_columns = list(df_train_feats.columns)
    #print(f"Features = {len(feature_columns)}")
    logger.info(f"Features = {len(feature_columns)}")
    #print(f"Feature length = {len(feature_columns)}")

    num_folds = 5
    fold_size = 480 // num_folds
    gap = 5

    models = []
    models_cbt = []
    scores = []


    model_save_path = os.path.join(base_dir, model_name, graph_name + '_modelitos_para_despues')
    logger.info(f"model_save_path {model_save_path}")

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    date_ids = df_train['date_id'].values

    now_time = time.time()
    time_cost_list = []

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
        
        gc.collect()
        
        df_fold_train = df_train_feats[train_indices]
        df_fold_train_target = df_train['target'][train_indices]
        df_fold_valid = df_train_feats[test_indices]
        df_fold_valid_target = df_train['target'][test_indices]

        logger.info(f"Fold {i+1} Model Training")

        # Train a LightGBM model for the current fold
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            df_fold_train[feature_columns],
            df_fold_train_target,
            eval_set=[(df_fold_valid[feature_columns], df_fold_valid_target)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=100),
            ],
        )
        
#         cbt_model = cbt.CatBoostRegressor(objective='MAE', iterations=5000,bagging_temperature=0.5,
#                                 colsample_bylevel = 0.7,learning_rate = 0.065,
#                                 od_wait = 25,max_depth = 7,l2_leaf_reg = 1.5,
#                                 min_data_in_leaf = 1000,random_strength=0.65,
#                                 verbose=0,use_best_model=True,task_type='CPU')
#         cbt_model.fit(
#             df_fold_train[feature_columns],
#             df_fold_train_target,
#             eval_set=[(df_fold_valid[feature_columns], df_fold_valid_target)]
#         )
        
#         models_cbt.append(cbt_model)

        time_cost = time.time() - now_time
        time_cost_list.append(time_cost)

        logger.info(f"cost time {time_cost}")

        models.append(lgb_model)
        # Save the model to a file
        model_filename = os.path.join(model_save_path, f'doblez_{i+1}.txt')
        lgb_model.booster_.save_model(model_filename)
        logger.info(f"Model for fold {i+1} saved to {model_filename}")

        # Evaluate model performance on the validation set
        #------------LGB--------------#
        fold_predictions = lgb_model.predict(df_fold_valid[feature_columns])
        fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
        scores.append(fold_score)
        logger.info(f":LGB Fold {i+1} MAE: {fold_score}")
        #------------CBT--------------#
#         fold_predictions = cbt_model.predict(df_fold_valid[feature_columns])
#         fold_score_cbt = mean_absolute_error(fold_predictions, df_fold_valid_target)
#         scores.append(fold_score_cbt)
#         print(f"CBT Fold {i+1} MAE: {fold_score_cbt}")

        # Free up memory by deleting fold specific variables
        del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
        gc.collect()

    # Calculate the average best iteration from all regular folds
    average_best_iteration = int(np.mean([model.best_iteration_ for model in models]))

    # Update the lgb_params with the average best iteration
    final_model_params = lgb_params.copy()

    # final_model_params['n_estimators'] = average_best_iteration
    # print(f"Training final model with average best iteration: {average_best_iteration}")

    # Train the final model on the entire dataset
    num_model = 1

    for i in range(num_model):
        final_model = lgb.LGBMRegressor(**final_model_params)
        final_model.fit(
            df_train_feats[feature_columns],
            df_train['target'],
            callbacks=[
                lgb.callback.log_evaluation(period=100),
            ],
        )
        # Append the final model to the list of models
        models.append(final_model)
        
        # Calculate and print the average MAE across all folds
LGB_average_mae = np.mean(scores)
time_cost_all = sum(time_cost_list)
logger.info(f"Average MAE across all folds: {LGB_average_mae}")
logger.info(f"Time cost all folds: {time_cost_all}")

## NN

def create_mlp(num_continuous_features, num_categorical_features, embedding_dims, num_labels, hidden_units, dropout_rates, learning_rate,l2_strength=0.01):

    # Numerical variables input
    input_continuous = tf.keras.layers.Input(shape=(num_continuous_features,))

    # Categorical variables input
    input_categorical = [tf.keras.layers.Input(shape=(1,))
                         for _ in range(len(num_categorical_features))]

    # Embedding layer for categorical variables
    embeddings = [tf.keras.layers.Embedding(input_dim=num_categorical_features[i],
                                            output_dim=embedding_dims[i])(input_cat)
                  for i, input_cat in enumerate(input_categorical)]
    flat_embeddings = [tf.keras.layers.Flatten()(embed) for embed in embeddings]

    # concat numerical and categorical
    concat_input = tf.keras.layers.concatenate([input_continuous] + flat_embeddings)

    # MLP
    x = tf.keras.layers.BatchNormalization()(concat_input)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)

    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i],kernel_regularizer=l2(0.01),kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        #x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i+1])(x)

    #No activation
    out = tf.keras.layers.Dense(num_labels,kernel_regularizer=l2(0.01),kernel_initializer='he_normal')(x)

    model = tf.keras.models.Model(inputs=[input_continuous] + input_categorical, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])
    return model

if NN:
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    import gc
    from sklearn.model_selection import KFold
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as layers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
    
    df_train_feats = df_train_feats.groupby('stock_id').apply(lambda group: group.fillna(method='ffill')).fillna(0)
    
    categorical_features = ["stock_id"]
    numerical_features = [column for column in list(df_train_feats) if column not in categorical_features]
    num_categorical_features = [len(df_train_feats[col].unique()) for col in categorical_features]

    nn_models = []

    batch_size = 64
    hidden_units = [128,128]
    dropout_rates = [0.1,0.1,0.1]
    learning_rate = 1e-5
    embedding_dims = [20]

    directory = os.path.join(base_dir, model_name, graph_name + 'NN_Models')
    if not os.path.exists(directory):
        os.mkdir(directory)

    pred = np.zeros(len(df_train['target']))
    scores = []
    gkf = PurgedGroupTimeSeriesSplit(n_splits = 5, group_gap = 5)


    for fold, (tr, te) in enumerate(gkf.split(df_train_feats,df_train['target'],df_train['date_id'])):

        ckp_path = os.path.join(directory, f'nn_Fold_{fold+1}.h5')

        X_tr_continuous = df_train_feats.iloc[tr][numerical_features].values
        X_val_continuous = df_train_feats.iloc[te][numerical_features].values

        X_tr_categorical = df_train_feats.iloc[tr][categorical_features].values
        X_val_categorical = df_train_feats.iloc[te][categorical_features].values

        y_tr, y_val = df_train['target'].iloc[tr].values, df_train['target'].iloc[te].values

        logger.info("X_train_numerical shape:",X_tr_continuous.shape)
        logger.info("X_train_categorical shape:",X_tr_categorical.shape)
        logger.info("Y_train shape:",y_tr.shape)
        logger.info("X_test_numerical shape:",X_val_continuous.shape)
        logger.info("X_test_categorical shape:",X_val_categorical.shape)
        logger.info("Y_test shape:",y_val.shape)

        logger.info(f"Creating Model - Fold{fold}")
        model = create_mlp(len(numerical_features), num_categorical_features, embedding_dims, 1, hidden_units, dropout_rates, learning_rate)

        rlr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1, patience=3, verbose=0, min_delta=1e-4, mode='min')
        ckp = ModelCheckpoint(ckp_path, monitor='val_mean_absolute_error', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
        es = EarlyStopping(monitor='val_mean_absolute_error', min_delta=1e-4, patience=10, mode='min', restore_best_weights=True, verbose=0)

        logger.info(f"Fitting Model - Fold{fold}")
        model.fit((X_tr_continuous,X_tr_categorical), y_tr,
                  validation_data=([X_val_continuous,X_val_categorical], y_val),
                  epochs=200, batch_size=batch_size,callbacks=[ckp,es,rlr])

        output = model.predict((X_val_continuous,X_val_categorical), batch_size=batch_size * 4)

        pred[te] += model.predict((X_val_continuous,X_val_categorical), batch_size=batch_size * 4).ravel()

        score = mean_absolute_error(y_val, pred[te])
        scores.append(score)
        logger.info(f'Fold {fold} MAE:\t', score)

        # Finetune 3 epochs on validation set with small learning rate
        logger.info(f"Finetuning Model - Fold{fold}")
        model = create_mlp(len(numerical_features), num_categorical_features, embedding_dims, 1, hidden_units, dropout_rates, learning_rate / 100)
        model.load_weights(ckp_path)
        model.fit((X_val_continuous,X_val_categorical), y_val, epochs=5, batch_size=batch_size, verbose=0)
        model.save_weights(ckp_path)
        nn_models.append(model)

        K.clear_session()
        del model
        gc.collect()
    NN_score = np.mean(scores)

    logger.info("Average NN CV Scores:",np.mean(scores))

### evaluation on test set

### Evaluation on Test Set
# Load Test Data
test_df = pd.read_csv("test.csv")  # Correct path to test.csv
test_df = test_df.dropna(subset=["target"])
test_df.reset_index(drop=True, inplace=True)

# Generate Features for Test Data
test_feats = generate_all_features(test_df)

# Reduce memory usage for Test Data (if needed)
test_feats = reduce_mem_usage(test_feats)

# Prepare Test Data for Neural Network
X_test_continuous = test_feats[numerical_features].values
X_test_categorical = test_feats[categorical_features].values

# Initialize array to hold test predictions
test_predictions = np.zeros(len(test_df))

# LightGBM Predictions
if LGB:
    for model in models:
        test_predictions += model.predict(test_feats[feature_columns]) / len(models)

# Neural Network Predictions
if NN:
    for model in nn_models:
        test_predictions += model.predict((X_test_continuous, X_test_categorical), batch_size=batch_size * 4).ravel() / len(nn_models)

# Averaging Predictions from both LGB and NN models if both are used
if LGB and NN:
    test_predictions /= 2

# Calculate Score
#if "target" in test_df.columns:
#    test_score = mean_absolute_error(test_df["target"], test_predictions)
#    logger.info(f"Test Mean Absolute Error: {test_score}")


# result
results = {"is_offline": is_offline, "LGB": LGB, "NN": NN,
            "is_train": is_train, "is_infer": is_infer, 
            "max_lookback": max_lookback, "split_day": split_day,
            "time_cost": time_cost_all,
            "Average NN CV Scores": NN_score, "LGB_average_mae": LGB_average_mae,
            "lgb_params": lgb_params}

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

with open(results_dir + "/{}.json".format(graph_name), 'w') as fout:
    fout.write(json.dumps(results))



if __name__ == '__main__':
    # set logger
    pass


