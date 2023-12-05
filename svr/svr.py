import gc
import os
import time
import warnings
from itertools import combinations
from warnings import simplefilter
import joblib
import numpy as np
import pandas as pd
#import polars as pl  # Replacing pandas with Polars
from cuml.svm import SVR  # Importing CUML's SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
warnings.filterwarnings("ignore")
#simplefilter(action="ignore", category=pl.errors.PerformanceWarning)  # Updated for Polars
is_offline = False
is_train = True
is_infer = True
max_lookback = np.nan
split_day = 435

import os  # Operating system-related functions
import time  # Time-related functions
import warnings  # Handling warnings
from itertools import combinations  # For creating combinations of elements
from warnings import simplefilter  # Simplifying warning handling

# 📦 Importing machine learning libraries
import joblib  # For saving and loading models
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
from sklearn.metrics import mean_absolute_error  # Metric for evaluation
from sklearn.model_selection import KFold, TimeSeriesSplit  # Cross-validation techniques

# 🤐 Disable warnings to keep the code clean
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# 📊 Define flags and variables
is_offline = False  # Flag for online/offline mode
is_train = True  # Flag for training mode
is_infer = True  # Flag for inference mode
max_lookback = np.nan  # Maximum lookback (not specified)
split_day = 435  # Split day for time series data

# input data
df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
df = df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df_shape = df.shape

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

from numba import njit, prange

# 优化代码，用以计算不平衡度
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


@njit(fastmath=True)
def rolling_average(arr, window):
    """
    Calculate the rolling average for a 1D numpy array.
    
    Parameters:
    arr (numpy.ndarray): Input array to calculate the rolling average.
    window (int): The number of elements to consider for the moving average.
    
    Returns:
    numpy.ndarray: Array containing the rolling average values.
    """
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan  # Padding with NaN for elements where the window is not full
    cumsum = np.cumsum(arr)

    for i in range(window, n):
        result[i] = (cumsum[i] - cumsum[i - window]) / window

    return result


@njit(parallel=True)
def compute_rolling_averages(df_values, window_sizes):
    """
    Calculate the rolling averages for multiple window sizes in parallel.
    
    Parameters:
    df_values (numpy.ndarray): 2D array of values to calculate the rolling averages.
    window_sizes (List[int]): List of window sizes for the rolling averages.
    
    Returns:
    numpy.ndarray: A 3D array containing the rolling averages for each window size.
    """
    num_rows, num_features = df_values.shape
    num_windows = len(window_sizes)
    rolling_features = np.empty((num_rows, num_features, num_windows))

    for feature_idx in prange(num_features):
        for window_idx, window in enumerate(window_sizes):
            rolling_features[:, feature_idx, window_idx] = rolling_average(df_values[:, feature_idx], window)

    return rolling_features

def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    # 因子挖掘
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

def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    # Generate imbalance features
    df = imbalance_features(df)
    df = other_features(df)
    gc.collect()  
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    
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

#     if is_offline:
#         df_train_feats = generate_all_features(df_train)
#         print("Build Train Feats Finished.")
#         df_valid_feats = generate_all_features(df_valid)
#         print("Build Valid Feats Finished.")
#         df_valid_feats = reduce_mem_usage(df_valid_feats)
#     else:
#         df_train_feats = generate_all_features(df_train)
#         print("Build Online Train Feats Finished.")

#     df_train_feats = reduce_mem_usage(df_train_feats)

import numpy as np
from cuml.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib
import gc
import os
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

# # Define SVR parameters (adjust these based on your dataset)
# svr_params = {
#     "kernel": "rbf",  # Example kernel type
#     "C": 1,           # Regularization parameter
#     "gamma": 'scale', # Kernel coefficient
#     # ... other SVR parameters as needed ...
# }

# feature_name = list(df_train_feats.columns)
# print(f"Feature length = {len(feature_name)}")

# num_folds = 5
# fold_size = 480 // num_folds
# gap = 5

# models = []
# scores = []

# model_save_path = 'modelitos_para_despues'
# if not os.path.exists(model_save_path):
#     os.makedirs(model_save_path)

# date_ids = df_train['date_id'].values

# for i in range(num_folds):
#     start = i * fold_size
#     end = start + fold_size
#     if i < num_folds - 1:  # No need to purge after the last fold
#         purged_start = end - 2
#         purged_end = end + gap + 2
#         train_indices = (date_ids >= start) & (date_ids < purged_start) | (date_ids > purged_end)
#     else:
#         train_indices = (date_ids >= start) & (date_ids < end)

#     test_indices = (date_ids >= end) & (date_ids < end + fold_size)
    
#     df_fold_train = df_train_feats[train_indices]
#     df_fold_train_target = df_train['target'][train_indices]
#     df_fold_valid = df_train_feats[test_indices]
#     df_fold_valid_target = df_train['target'][test_indices]

#     # Impute NaN values
#     df_fold_train_imputed = imputer.fit_transform(df_fold_train[feature_name])
#     df_fold_valid_imputed = imputer.transform(df_fold_valid[feature_name])
    
#     print(f"Fold {i+1} Model Training")
    
#     # Train an SVR model for the current fold
#     svr_model = SVR(**svr_params)
#     svr_model.fit(df_fold_train_imputed, df_fold_train_target)

#     models.append(svr_model)
    
#     # Save the model to a file using joblib
#     model_filename = os.path.join(model_save_path, f'svr_model_fold_{i+1}.joblib')
#     joblib.dump(svr_model, model_filename)

#     print(f"Model for fold {i+1} saved to {model_filename}")

#     # Evaluate model performance on the validation set
#     fold_predictions = svr_model.predict(df_fold_valid_imputed)
#     fold_score = mean_absolute_error(df_fold_valid_target, fold_predictions)
#     scores.append(fold_score)
#     print(f"Fold {i+1} MAE: {fold_score}")

#     # Free up memory
#     del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target, df_fold_valid_imputed, df_fold_train_imputed
#     gc.collect()

# print(f"Training final model")

# # Train the final model on the entire dataset
# final_model = SVR(**svr_params)
# final_model.fit(df_train_feats[feature_name], df_train['target'])

# # Save the final model to a file
# final_model_filename = os.path.join(model_save_path, 'final_svr_model.joblib')
# joblib.dump(final_model, final_model_filename)
# print(f"Final model saved to {final_model_filename}")
# print(f"Average MAE across all folds: {np.mean(scores)}")

import joblib
# Load the model from the file
final_model = joblib.load("/kaggle/input/svr-models/svr_model_fold_1.joblib")
models = [final_model]

# Load other models that you may want to use for an ensemble
import os
import lightgbm as lgb

def load_models_from_folder(model_save_path, num_folds=5):
    loaded_models = []

    # Load each fold model
    for i in range(1, num_folds + 1):
        model_filename = os.path.join(model_save_path, f'doblez_{i}.txt')
        if os.path.exists(model_filename):
            loaded_model = lgb.Booster(model_file=model_filename)
            loaded_models.append(loaded_model)
            print(f"Model for fold {i} loaded from {model_filename}")
        else:
            print(f"Model file {model_filename} not found.")

    # Load the final model
    final_model_filename = os.path.join(model_save_path, 'doblez-conjunto.txt')
    if os.path.exists(final_model_filename):
        final_model = lgb.Booster(model_file=final_model_filename)
        loaded_models.append(final_model)
        print(f"Final model loaded from {final_model_filename}")
    else:
        print(f"Final model file {final_model_filename} not found.")
        return loaded_models

# Assuming you have a list of folders from which to load the models
folders = [
#     '/kaggle/input/lightgbm-models/modelitos_para_despues',
    '/kaggle/input/ensemble-of-models/results/modelitos_para_despues',
    '/kaggle/input/ensemble-of-models/results (1)/modelitos_para_despues',
    '/kaggle/input/ensemble-of-models/results (2)/modelitos_para_despues',
    '/kaggle/input/ensemble-of-models/results (3)/modelitos_para_despues',
#      '/kaggle/input/ensemble-of-models/results (4)/modelitos_para_despues',
#      '/kaggle/input/ensemble-of-models/results (5)/modelitos_para_despues',
#     '/kaggle/input/ensemble-of-models/results (6)/modelitos_para_despues',
#     '/kaggle/input/ensemble-of-models/results (7)/modelitos_para_despues',
]
num_folds = 5
all_loaded_models = []
for folder in folders:
    all_loaded_models.extend(load_models_from_folder(folder))

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
    model_weights_lgbm = [1 / len(all_loaded_models)] * len(all_loaded_models)
    model_weights = [1 / len(models)] * len(models)
    
    for (test, revealed_targets, sample_prediction) in iter_test:
        try:
            test = test.drop('currently_scored', axis = 1)
            now_time = time.time()
            cache = pd.concat([cache, test], ignore_index=True, axis=0)
            if counter > 0:
                cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
            feat = generate_all_features(cache)[-len(test):]
            pred = np.zeros(len(test))
            # Supportive models
            for model, weight in zip(all_loaded_models, model_weights_lgbm):
                pred += weight * model.predict(feat)
            # Generate predictions for each model and calculate the weighted average
            svr_predictions = np.zeros(len(test))
            for model, weight in zip(models, model_weights):
                svr_predictions += weight * model.predict(feat.fillna(0.0))
                
            svr_predictions = [pred[i] if np.isnan(x) else x for i, x in enumerate(svr_predictions)]
            svr_predictions = (len(models) * svr_predictions + len(all_loaded_models) * pred) / (len(models) + len(all_loaded_models))
            
            svr_predictions = zero_sum(svr_predictions, test['bid_size'] + test['ask_size'])
            clipped_predictions = np.clip(svr_predictions, y_min, y_max)
            # Replace NaN values with zeros
            sample_prediction['target'] = clipped_predictions
            env.predict(sample_prediction)
        except:
            sample_prediction['target'] = 0.0
            env.predict(sample_prediction)
        counter += 1
        qps.append(time.time() - now_time)
        if counter % 10 == 0:
            print(counter, 'qps:', np.mean(qps))

    time_cost = 1.146 * np.mean(qps)
    print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")





