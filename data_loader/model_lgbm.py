import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
import joblib
from lightgbm import LGBMRegressor, plot_importance
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import params

# Dataset 불러오기
data = params.DATA_SET
# load
with open(data, 'rb') as f:
    vector_dataset = pickle.load(f)


X_dataset = vector_dataset.iloc[:,:params.ANALYSIS_DAY].reset_index(drop=True)
Y_dataset = vector_dataset.iloc[:,params.ANALYSIS_DAY*2:].reset_index(drop=True)
# random_seed 설정
SEED = 1

# Train Vaild Test split
X_train, X_vaild, Y_train, Y_vaild = train_test_split(X_dataset, Y_dataset, test_size=0.1, random_state=SEED, shuffle=True)

#===========================================================================================
mae_vaild_ls = []
mae_test_ls = []
param_ls = []


optuna == LGBMRegressor
sampler = TPESampler(seed=10)
def objective(trial):
    param = {
        'objective': 'regression', # 회귀
        'verbose': 1,
        'metric': 'mae',
        # 'boosting': trial.suggest_categorical("boosting", ["gbdt", "dart", "goss"]),
        # 무한으로 분기할것!
        # 'max_depth': trial.suggest_int('max_depth',3, 20),
        'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-2),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'num_iterations' : trial.suggest_int('num_iterations', 1000, 1500),
        'scale_pos_weight' : trial.suggest_float('scale_pos_weight',1.0,1.5)
    }
    model = LGBMRegressor(**param)
    lgb_model = model.fit(X_train, Y_train.values.ravel(), eval_set=[(X_vaild, Y_vaild)], verbose=2)
    mae = mean_absolute_error(Y_vaild, lgb_model.predict(X_vaild))
    # mae_test = mean_absolute_error(Y_test, lgb_model.predict(X_test))
    param_ls.append(lgb_model.get_params)  
    mae_vaild_ls.append(mae)
    # mae_test_ls.append(mae_test)
    return mae

study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=20)


print(study.best_params)
final_lgb_model = LGBMRegressor(boosting ="dart", max_depth= -1, learning_rate= study.best_params['learning_rate'], subsample= study.best_params['subsample'],num_iterations=study.best_params['num_iterations'],scale_pos_weight=study.best_params['scale_pos_weight'])
final_lgb_model.fit(X_train, Y_train)

pred_vaild = final_lgb_model.predict(X_vaild)
vaild_mae = mean_absolute_error(Y_vaild,pred_vaild)

# print(vaild_mae, test_mae)
# 모델 저장
joblib.dump(final_lgb_model, f"../data/lgbm_model_{vaild_mae}.pkl") 
#===========================================================================================
