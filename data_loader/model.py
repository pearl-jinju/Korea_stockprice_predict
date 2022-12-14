import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
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




X_dataset = vector_dataset.iloc[:,:params.ANALYSIS_DAY].reset_index(drop=True)
Y_dataset = vector_dataset.iloc[:,params.ANALYSIS_DAY*2:].reset_index(drop=True)

scaler = MinMaxScaler()
X_dataset = scaler.fit_transform(X_dataset)
print(X_dataset)

# random_seed 설정
SEED = 1234

# Train Vaild Test split
X_train, X_vaild, Y_train, Y_vaild = train_test_split(X_dataset, Y_dataset, test_size=0.2, random_state=SEED, shuffle=True)
X_vaild, X_test, Y_vaild, Y_test = train_test_split(X_vaild, Y_vaild, test_size=0.5, random_state=SEED, shuffle=True)

mae_vaild_ls = []
mae_test_ls = []
param_ls = []

# =======================================================================================
optuna == LGBMRegressor
sampler = TPESampler(seed=10)
def objective(trial):
    param = {
        'objective': 'regression', # 회귀
        'verbose': 1,
        'metric': 'mae',
        'boosting': trial.suggest_categorical("boosting", ["gbdt", "dart", "goss"]),
        'max_depth': trial.suggest_int('max_depth',3, 20),
        'learning_rate': trial.suggest_float("learning_rate", 1e-9, 1e-2),
        # 'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'num_iterations' : trial.suggest_int('num_iterations', 1000, 1200),
        'scale_pos_weight' : trial.suggest_float('scale_pos_weight',1.1,1.5)
    }
    model = LGBMRegressor(**param)
    lgb_model = model.fit(X_train, Y_train.values.ravel(), eval_set=[(X_vaild, Y_vaild)], verbose=0)
    mae = mean_absolute_error(Y_vaild, lgb_model.predict(X_vaild))
    mae_test = mean_absolute_error(Y_test, lgb_model.predict(X_test))
    param_ls.append(lgb_model.get_params)  
    mae_vaild_ls.append(mae)
    mae_test_ls.append(mae_test)
    return mae_test

study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=20)


print(study.best_params)
final_lgb_model = LGBMRegressor(boosting =study.best_params['boosting'], max_depth= study.best_params['max_depth'], learning_rate= study.best_params['learning_rate'], subsample= study.best_params['subsample'],num_iterations=study.best_params['num_iterations'],scale_pos_weight=study.best_params['scale_pos_weight'])
# final_lgb_model = LGBMRegressor(boosting =study.best_params['boosting'], max_depth= study.best_params['max_depth'], learning_rate= study.best_params['learning_rate'], n_estimators= study.best_params['n_estimators'], subsample= study.best_params['subsample'],num_iterations=study.best_params['num_iterations'],scale_pos_weight=study.best_params['scale_pos_weight'])
final_lgb_model.fit(X_train, Y_train)

pred_vaild = final_lgb_model.predict(X_vaild)
vaild_mae = mean_absolute_error(Y_vaild,pred_vaild)
pred_test = final_lgb_model.predict(X_test)
test_mae = mean_absolute_error(Y_test,pred_test)

print(vaild_mae, test_mae)
# 모델 저장
joblib.dump(final_lgb_model, "../data/lgbm_model.pkl") 
# =======================================================================================


# # optuna == SVR  현재 미작동 코드 수정 필요
# sampler = TPESampler(seed=10)
# def objective(trial):
#     param = {
#         'C': trial.suggest_float('C', 0.3, 10.0),
#         'degree': trial.suggest_int('degree', 2, 7),
#         'gamma': trial.suggest_float('gamma', 1e-3, 0.01),
#         'epsilon': trial.suggest_float('epsilon', 0.05, 0.5),
#     }
#     clf = SVR(**param)
#     clf.fit(X_train, Y_train.values.ravel())
#     mae = mean_absolute_error(Y_vaild, clf.predict(X_vaild))
#     mae_test = mean_absolute_error(Y_test, clf.predict(X_test))
#     param_ls.append(lgb_model.get_params)  
#     mae_vaild_ls.append(mae)
#     mae_test_ls.append(mae_test)
#     return mae_test

# study = optuna.create_study(direction='minimize', sampler=sampler)
# study.optimize(objective, n_trials=1)

# print(study.best_params)

# final_svr_model = SVR(C =study.best_params['C'], degree= study.best_params['degree'], gamma= study.best_params['gamma'], epsilon= study.best_params['epsilon'])
# final_svr_model.fit(X_train, Y_train.values.ravel())

# pred_vaild = final_svr_model.predict(X_vaild)
# vaild_mae = mean_absolute_error(Y_vaild,pred_vaild)
# pred_test = final_svr_model.predict(X_test)
# test_mae = mean_absolute_error(Y_test,pred_test)

# print(vaild_mae, test_mae)
# # 모델 저장
# joblib.dump(final_svr_model, "../data/final_svr_model.pkl") 


# plt.plot(range(20),mae_vaild_ls, color='b')
# plt.plot(range(20),mae_test_ls, marker='+', color='r')
# plt.title('vaild_test', fontsize=20) 
# plt.ylabel('mae', fontsize=14)
# plt.xlabel('n_trials', fontsize=14)
# plt.show()

# params_df = pd.DataFrame(param_ls)
# params_df.to_csv('Best_params.csv', index= False)

# plot_importance 시각화
# fm.get_fontconfig_fonts()
# FONT_LOCATION = r'C:\Windows\Fonts/malgun.ttf'
# font_name = fm.FontProperties(fname=FONT_LOCATION).get_name()
# plt.rc('font', family=font_name)
# print(plt.rcParams['font.family'])
# fig, ax = plt.subplots(figsize=(10,12))
# plot_importance(lgbm_wrapper, ax=ax)
# plt.show()