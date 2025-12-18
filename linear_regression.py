import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# ==========================================
# 1. 데이터 로드 (압축된 데이터셋 사용)
# ==========================================
print("--- [1/5] 데이터 로드 ---")
#compressed_filename = "dataset/ml_dataset.csv"
# 만약 hidden_size 버전이라면 아래 주석 해제
#compressed_filename = "dataset/ml_dataset_hidden_size.csv"
compressed_filename = "dataset/ml_dataset_memory_size.csv"

try:
    df = pd.read_csv(compressed_filename)
    print(f"  - '{compressed_filename}' 로드 성공 ({len(df)} rows)")
except FileNotFoundError:
    print(f"오류: '{compressed_filename}' 파일을 찾을 수 없습니다.")
    exit()

output_columns = [
    'fwd_avg_w', 'fwd_max_w', 'fwd_min_w',
    'comm_avg_w', 'comm_max_w', 'comm_min_w',
    'bwd_avg_w', 'bwd_max_w', 'bwd_min_w'
]

# 입력 피처 (데이터셋에 있는 컬럼만 자동 선택)
potential_features = [
    'network_bandwidth', 'gpu_freq', 'cpu_freq',
    'active_gpu_cores', 'active_cpu_cores', 'transformer_blocks',
    'batch_size' # hidden_size가 있다면 포함됨
]
input_features = [col for col in potential_features if col in df.columns]
print(f"  - 사용된 Input Features: {input_features}")

X = df[input_features]
Y = df[output_columns]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Data Prepared. Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 2. 선형 회귀 모델 학습
# ==========================================
print("\n--- [2/5] 선형 회귀 모델 학습 ---")

# 선형 회귀는 하이퍼파라미터 튜닝이 거의 필요 없습니다.
model = LinearRegression()
multi_model = MultiOutputRegressor(model)

multi_model.fit(X_train, Y_train)
print("✅ 학습 완료.")

# ==========================================
# 3. 모델 평가
# ==========================================
print("\n--- [3/5] 최종 모델 평가 ---")

Y_pred = multi_model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(Y_test, Y_pred)

print(f"  [Overall] MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape*100:.2f}%")

print("\n  [Detailed Performance by Output]")
mae_per_col = []
for i, col in enumerate(output_columns):
    y_true_col = Y_test.iloc[:, i]
    y_pred_col = Y_pred[:, i]
    
    c_mae = mean_absolute_error(y_true_col, y_pred_col)
    c_rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
    c_mape = mean_absolute_percentage_error(y_true_col, y_pred_col)
    
    mae_per_col.append(c_mae)
    print(f"    - {col:<12}: MAPE={c_mape*100:.2f}%")

# ==========================================
# 4. 결과 시각화 (그래프 3종)
# ==========================================
print("\n--- [4/5] 결과 시각화 및 저장 ---")
plt.style.use('seaborn-v0_8-whitegrid')

# [Graph 1] MAE Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=mae_per_col, y=output_columns, palette='viridis', hue=output_columns, legend=False)
plt.xlabel('MAE (Watt)', fontsize=14, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
for i, v in enumerate(mae_per_col): 
    plt.text(v + 0.01, i, f"{v:.4f}", color='black', va='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("eval/linear/linear_result_1_mae_bar.png", dpi=300)

# [Graph 2] Overall Scatter
plt.figure(figsize=(10, 10))
y_true_f, y_pred_f = Y_test.values.flatten(), Y_pred.flatten()
plt.scatter(y_true_f, y_pred_f, alpha=0.2, s=15, color='darkgreen') # 선형회귀는 녹색 계열
plt.plot([min(y_true_f), max(y_true_f)], [min(y_true_f), max(y_true_f)], 'r--', lw=2.5)
plt.xlabel('True Values (W)', fontsize=16, fontweight='bold')
plt.ylabel('Predicted Values (W)', fontsize=16, fontweight='bold')
stats_text = (f'MAPE: {mape*100:.2f}%')
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=18, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
plt.tight_layout()
plt.savefig("eval/linear/linear_result_2_scatter_overall.png", dpi=300)

# [Graph 3] Detailed Scatter (3x3)
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()
for i, col in enumerate(output_columns):
    yt, yp = Y_test.iloc[:, i], Y_pred[:, i]
    c_mae = mean_absolute_error(yt, yp)
    c_rmse = np.sqrt(mean_squared_error(yt, yp))
    c_mape = mean_absolute_percentage_error(yt, yp) * 100
    
    axes[i].scatter(yt, yp, alpha=0.3, s=20, color='seagreen')
    axes[i].plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--', lw=2)
    axes[i].set_xlabel('True (W)', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Pred (W)', fontsize=12, fontweight='bold')
    
    label_text = (f"{col}\nMAE: {c_mae:.2f}\nRMSE: {c_rmse:.2f}\nMAPE: {c_mape:.1f}%")
    axes[i].text(0.05, 0.95, label_text, transform=axes[i].transAxes, fontsize=12, fontweight='bold',
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig("eval/linear/linear_result_3_scatter_detailed.png", dpi=300)

# ==========================================
# 5. 가중치(Coefficient) 분석 (선형 회귀의 핵심!)
# ==========================================
print("\n--- [5/5] Feature Coefficients 분석 (영향력 확인) ---")

# 각 타겟별로 학습된 계수(Coefficient)를 추출
coef_data = []
for i, col in enumerate(output_columns):
    estimator = multi_model.estimators_[i]
    coefs = estimator.coef_
    for feat, coef in zip(input_features, coefs):
        coef_data.append({'Target': col, 'Feature': feat, 'Coefficient': coef})

coef_df = pd.DataFrame(coef_data)

# 히트맵으로 시각화 (어떤 변수가 전력을 올리고 내리는지 한눈에 파악)
plt.figure(figsize=(12, 8))
coef_pivot = coef_df.pivot(index='Target', columns='Feature', values='Coefficient')
sns.heatmap(coef_pivot, annot=True, fmt=".2e", cmap="coolwarm", center=0)
plt.title('Linear Regression Coefficients (Weight)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("eval/linear/linear_result_4_coefficients.png", dpi=300)

# 모델 저장
joblib.dump(multi_model, "model/linear/linear_regression_model.pkl")
print(f"\n✅ 모델 저장 완료: model/linear/linear_regression_model.pkl")
print("✅ 모든 그래프 저장 완료.")

# ==========================================
# 6. 평가 결과 CSV 저장 (for Prism)
# ==========================================
print("\n--- [6/5] 평가 결과 CSV 저장 (for Prism) ---")

# 결과 저장을 위한 DataFrame 생성
eval_results = pd.DataFrame()

# 실제값과 예측값을 컬럼별로 저장
for i, col in enumerate(output_columns):
    # 실제값
    if hasattr(Y_test, 'iloc'):
        eval_results[f'True_{col}'] = Y_test.iloc[:, i].values
    else:
        eval_results[f'True_{col}'] = Y_test[:, i]
        
    # 예측값
    eval_results[f'Pred_{col}'] = Y_pred[:, i]

# 파일 저장
save_csv_name = "eval/linear/linear_evaluation_results.csv"
eval_results.to_csv(save_csv_name, index=False)
print(f"✅ 평가 결과가 '{save_csv_name}' 파일로 저장되었습니다.")