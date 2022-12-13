import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from lightgbm import LGBMRegressor, plot_importance
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import params


data = params.DATA_SET
# load
with open(data, 'rb') as f:
    vector_dataset = pickle.load(f)

X_dataset = vector_dataset.iloc[:,:20].reset_index(drop=True)
Y_dataset = vector_dataset.iloc[:,40:].reset_index(drop=True)

# random_seed 설정
SEED = 42 

# Train Vaild Test split
X_train, X_vaild, Y_train, Y_vaild = train_test_split(X_dataset, Y_dataset, test_size=0.2, random_state=SEED)
X_vaild, X_test, Y_vaild, Y_test = train_test_split(X_vaild, Y_vaild, test_size=0.5, random_state=SEED)


# # optuna 모델을 뽑을때만 사용
# sampler = TPESampler(seed=10)
# def objective(trial):
#     param = {
#         'objective': 'regression', # 회귀
#         'verbose': -1,
#         'metric': 'mae', 
#         'max_depth': trial.suggest_int('max_depth',2, 99),
#         'learning_rate': trial.suggest_loguniform("learning_rate", 1e-9, 1e-2),
#         'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
#         'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
#         'subsample': trial.suggest_loguniform('subsample', 0.1, 1),
#     }
#     model = LGBMRegressor(**param)
#     lgb_model = model.fit(X_train, Y_train, eval_set=[(X_vaild, Y_vaild)], verbose=0)
#     mae = mean_absolute_error(Y_vaild, lgb_model.predict(X_vaild))
#     return mae
        
# study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
# study_lgb.optimize(objective, n_trials=50)


# print(study_lgb.best_params)

params = {'max_depth': 16, 'learning_rate': 0.009735849026134495, 'n_estimators': 371, 'min_child_samples': 57, 'subsample': 0.9902656518810361}
final_lgb_model = LGBMRegressor(max_depth= params['max_depth'], learning_rate= params['learning_rate'], n_estimators= params['n_estimators'], min_child_samples= params['min_child_samples'], subsample= params['subsample'])
final_lgb_model.fit(X_train, Y_train)

pred_vaild = final_lgb_model.predict(X_vaild)
vaild_mae = mean_absolute_error(Y_vaild,pred_vaild)
pred_test = final_lgb_model.predict(X_test)
test_mae = mean_absolute_error(Y_test,pred_test)


print(vaild_mae, test_mae)
# 모델 저장
joblib.dump(final_lgb_model, "../data/lgbm_model.pkl") 


# plot_importance 시각화
# fm.get_fontconfig_fonts()
# FONT_LOCATION = r'C:\Windows\Fonts/malgun.ttf'
# font_name = fm.FontProperties(fname=FONT_LOCATION).get_name()
# plt.rc('font', family=font_name)
# print(plt.rcParams['font.family'])
# fig, ax = plt.subplots(figsize=(10,12))
# plot_importance(lgbm_wrapper, ax=ax)
# plt.show()