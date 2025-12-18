import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import time
import joblib

# ==========================================
# 1. 데이터 로드 (압축된 데이터셋 사용)
# ==========================================
print("--- [1/5] 데이터 로드 ---")
#compressed_filename = "dataset/ml_dataset.csv"
compressed_filename = "dataset/ml_dataset_hidden_size.csv"
#compressed_filename = "dataset/ml_dataset_memory_size.csv"
#compressed_filename = "dataset/ml_dataset_kv_cache.csv"

try:
    df = pd.read_csv(compressed_filename)
    print(f"  - '{compressed_filename}' 로드 성공 ({len(df)} rows)")
except FileNotFoundError:
    print(f"오류: '{compressed_filename}' 파일을 찾을 수 없습니다. parse.py를 먼저 실행하세요.")
    exit()

output_columns = [
    'fwd_avg_w', 'fwd_max_w', 'fwd_min_w',
    'comm_avg_w', 'comm_max_w', 'comm_min_w',
    'bwd_avg_w', 'bwd_max_w', 'bwd_min_w'
]

input_features = [
    'network_bandwidth', 'gpu_freq', 'cpu_freq', 
    #'kv_cache', 
    'hidden_size', 
    #'memory_limit_gb',
    'active_gpu_cores', 'active_cpu_cores', 'transformer_blocks'
]

X = df[input_features]
Y = df[output_columns]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Data Prepared. Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 2. Optuna 하이퍼파라미터 최적화
# ==========================================
print("\n--- [2/5] Optuna 하이퍼파라미터 최적화 시작 (MAE 기준) ---")

def objective(trial):
    params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'device': 'cuda', 
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
        'n_jobs': -1,
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    multi_model = MultiOutputRegressor(model)
    multi_model.fit(X_train, Y_train)
    preds = multi_model.predict(X_test)
    return mean_absolute_error(Y_test, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f"🏆 Best MAE: {study.best_value:.4f}")

# ==========================================
# 3. 최적 파라미터로 최종 모델 학습
# ==========================================
print("\n--- [3/5] 최적 파라미터로 최종 모델 학습 ---")

best_params = study.best_params
best_params.update({'objective': 'reg:absoluteerror', 'tree_method': 'hist', 'device': 'cuda', 'n_jobs': -1, 'random_state': 42})

final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params))
start_time = time.time()
final_model.fit(X_train, Y_train)
print(f"Training Complete! (Time: {time.time() - start_time:.2f} sec)")

# ==========================================
# 4. 모델 평가
# ==========================================
print("\n--- [4/5] 최종 모델 평가 ---")

Y_pred = final_model.predict(X_test)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(Y_test, Y_pred)

print(f"  [Overall] MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape*100:.2f}%")

print("\n  [Detailed Performance by Output]")
mae_per_col = []
for i, col in enumerate(output_columns):
    y_true, y_pred_col = Y_test.iloc[:, i], Y_pred[:, i]
    c_mae = mean_absolute_error(y_true, y_pred_col)
    c_rmse = np.sqrt(mean_squared_error(y_true, y_pred_col))
    c_mape = mean_absolute_percentage_error(y_true, y_pred_col)
    mae_per_col.append(c_mae)
    print(f"    - {col:<12}: MAE={c_mae:.4f}, RMSE={c_rmse:.4f}, MAPE={c_mape*100:.2f}%")

# ==========================================
# 5. 결과 시각화 (제목 삭제 및 MAPE 추가)
# ==========================================
print("\n--- [5/5] 결과 시각화 및 저장 ---")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. MAE Bar (제목 삭제)
plt.figure(figsize=(12, 6))
sns.barplot(x=mae_per_col, y=output_columns, palette='viridis', hue=output_columns, legend=False)
# plt.title(...)  <-- 삭제됨
plt.xlabel('MAE (Watt)', fontsize=14, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
for i, v in enumerate(mae_per_col): 
    plt.text(v + 0.01, i, f"{v:.4f}", color='black', va='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("eval/xgboost/result_1_mae_bar.png", dpi=300)


# 2. Overall Scatter (제목 삭제, 내부에 MAPE 포함 정보 표시)
plt.figure(figsize=(10, 10))
y_true_f, y_pred_f = Y_test.values.flatten(), Y_pred.flatten()
plt.scatter(y_true_f, y_pred_f, alpha=0.2, s=15, color='darkblue')
plt.plot([min(y_true_f), max(y_true_f)], [min(y_true_f), max(y_true_f)], 'r--', lw=2.5)

# plt.title(...) <-- 삭제됨
plt.xlabel('True Values (W)', fontsize=16, fontweight='bold')
plt.ylabel('Predicted Values (W)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 텍스트 박스 추가 (MAE, RMSE, MAPE)
stats_text = (f'MAPE: {mape*100:.2f}%')
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=18, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.savefig("eval/xgboost/result_2_scatter_overall.png", dpi=300)


# 3. Detailed Scatter (제목 삭제, 내부에 컬럼명 및 MAPE 포함 표시)
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()
for i, col in enumerate(output_columns):
    yt, yp = Y_test.iloc[:, i], Y_pred[:, i]
    c_mae = mean_absolute_error(yt, yp)
    c_rmse = np.sqrt(mean_squared_error(yt, yp))
    c_mape = mean_absolute_percentage_error(yt, yp) * 100
    
    axes[i].scatter(yt, yp, alpha=0.3, s=20, color='royalblue')
    axes[i].plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--', lw=2)
    
    # axes[i].set_title(...) <-- 삭제됨
    axes[i].set_xlabel('True (W)', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Pred (W)', fontsize=12, fontweight='bold')
    
    # 내부 텍스트 라벨 (컬럼명 + 지표)
    label_text = (f"{col}\n"
                  f"MAPE: {c_mape:.1f}%")
    
    axes[i].text(0.05, 0.95, label_text, transform=axes[i].transAxes, fontsize=12, fontweight='bold',
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("eval/xgboost/result_3_scatter_detailed.png", dpi=300)


# 4. Feature Importance (제목 삭제)
avg_importances = np.mean([est.feature_importances_ for est in final_model.estimators_], axis=0)
fi_df = pd.DataFrame({'Feature': input_features, 'Importance': avg_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='rocket')
# plt.title(...) <-- 삭제됨
plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
plt.ylabel('Input Features', fontsize=14, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
for i, v in enumerate(fi_df['Importance']):
    plt.text(v + 0.001, i, f"{v:.4f}", color='black', va='center', fontsize=12)
plt.tight_layout()
plt.savefig("eval/xgboost/result_4_feature_importance.png", dpi=300)

print("✅ 모든 결과 저장 완료 (제목 삭제 및 MAPE 추가 적용됨).")

# ==========================================
# 7. 모델 저장 (Save Model)
# ==========================================
print("\n--- [6/6] 모델 저장 ---")
model_filename = "model/xgboost/xgboost_power_model_hidden_size.pkl"
joblib.dump(final_model, model_filename)
print(f"✅ 학습된 모델이 '{model_filename}' 파일로 저장되었습니다.")
print("   이제 'inference.py'를 통해 새로운 데이터의 전력을 예측할 수 있습니다.")

# ==========================================
# 7. 평가 결과 CSV 저장 (for Prism)
# ==========================================
print("\n--- [7/6] 평가 결과 CSV 저장 (for Prism) ---")

eval_results = pd.DataFrame()

for i, col in enumerate(output_columns):
    # XGBoost Y_test는 DataFrame이므로 iloc 사용
    eval_results[f'True_{col}'] = Y_test.iloc[:, i].values
    eval_results[f'Pred_{col}'] = Y_pred[:, i] # Y_pred는 numpy array

save_csv_name = "eval/xgboost/xgboost_evaluation_results.csv"
eval_results.to_csv(save_csv_name, index=False)
print(f"✅ 평가 결과가 '{save_csv_name}' 파일로 저장되었습니다.")