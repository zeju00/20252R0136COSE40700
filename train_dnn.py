import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna  # Optuna 추가

# ==========================================
# 1. 데이터 로드 및 전처리 (기존 코드 동일)
# ==========================================
print("--- [1/6] 데이터 로드 및 전처리 ---")

try:
    #df = pd.read_csv("dataset/ml_dataset.csv")
    #df = pd.read_csv("dataset/ml_dataset_hidden_size.csv")
    df = pd.read_csv("dataset/ml_dataset_memory_size.csv")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다.")
    exit()

output_columns = [
    'fwd_avg_w', 'fwd_max_w', 'fwd_min_w',
    'comm_avg_w', 'comm_max_w', 'comm_min_w',
    'bwd_avg_w', 'bwd_max_w', 'bwd_min_w'
]

input_features = [
    'network_bandwidth', 'gpu_freq', 'cpu_freq',
    'active_gpu_cores', 'active_cpu_cores', 'transformer_blocks'
]

df_cleaned = df.dropna(subset=output_columns)
df_grouped = df_cleaned.groupby(input_features)[output_columns].mean().reset_index()

X = df_grouped[input_features]
Y = df_grouped[output_columns]

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 스케일링
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
Y_train_scaled = y_scaler.fit_transform(Y_train)
X_test_scaled = x_scaler.transform(X_test)
Y_test_values = Y_test.values # 평가용 원본 값

joblib.dump(x_scaler, 'model/dnn/dnn_x_scaler.pkl')
joblib.dump(y_scaler, 'model/dnn/dnn_y_scaler.pkl')

# 텐서 변환
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
Y_train_tensor = torch.FloatTensor(Y_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

input_dim = len(input_features)
output_dim = len(output_columns)

print(f"Data Setup Complete. Device: {device}")

# ==========================================
# 2. Optuna Objective 함수 정의
# ==========================================
def build_model(trial, input_dim, output_dim):
    """Optuna Trial에 따라 동적으로 모델 구조를 생성하는 함수"""
    n_layers = trial.suggest_int("n_layers", 1, 4) # 은닉층 1~4개 탐색
    layers = []
    
    in_features = input_dim
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 32, 256) # 노드 수 32~256 탐색
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        
        # Dropout 적용 (과적합 방지)
        p = trial.suggest_float(f"dropout_l{i}", 0.0, 0.5)
        layers.append(nn.Dropout(p))
        
        in_features = out_features
        
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)

def objective(trial):
    # 1. 모델 생성
    model = build_model(trial, input_dim, output_dim).to(device)
    
    # 2. 하이퍼파라미터 제안
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    
    # Optimizer 설정
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        
    criterion = nn.L1Loss() # MAE Loss

    # DataLoader 생성 (Batch Size가 튜닝 대상이므로 여기서 생성)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. 학습 루프 (Pruning 포함)
    model.train()
    for epoch in range(50): # 튜닝용 Epoch은 조금 짧게 설정 (속도 위해)
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation (Pruning 용)
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_test_tensor)
            val_loss = criterion(outputs_val, torch.FloatTensor(y_scaler.transform(Y_test)).to(device)).item()
        model.train()
        
        # Optuna에게 현재 성능 보고
        trial.report(val_loss, epoch)
        
        # 가망 없으면 조기 종료 (Pruning)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

# ==========================================
# 3. Optuna 최적화 실행
# ==========================================
print("\n--- [2/6] Optuna Hyperparameter Tuning ---")
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100) # 50번 시도 (시간에 따라 조절)

print("\n🏆 Best Trial:")
print(f"  Value (Scaled MAE): {study.best_value:.4f}")
print("  Params: ")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# ==========================================
# 4. 최적 파라미터로 최종 재학습 (Full Epochs)
# ==========================================
print("\n--- [3/6] Retraining with Best Parameters ---")

best_params = study.best_params
final_epochs = 200 # 최종 학습은 충분히 길게

# Best Model 구조 재생성
# 주의: build_model은 trial 객체를 필요로 하므로, best_params 딕셔너리를 활용해 수동으로 구성하거나
# Optuna의 FixedTrial을 쓸 수 있지만, 여기선 직관적으로 다시 구성합니다.

layers = []
in_features = input_dim
for i in range(best_params['n_layers']):
    out_features = best_params[f"n_units_l{i}"]
    layers.append(nn.Linear(in_features, out_features))
    layers.append(nn.ReLU())
    p = best_params[f"dropout_l{i}"]
    layers.append(nn.Dropout(p))
    in_features = out_features
layers.append(nn.Linear(in_features, output_dim))

best_model = nn.Sequential(*layers).to(device)

# Optimizer 재설정
lr = best_params['lr']
if best_params['optimizer'] == "Adam":
    optimizer = optim.Adam(best_model.parameters(), lr=lr)
else:
    optimizer = optim.RMSprop(best_model.parameters(), lr=lr)

criterion = nn.L1Loss()
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

# 최종 학습
best_model.train()
loss_history = []

for epoch in range(final_epochs):
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch [{epoch+1}/{final_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# 모델 저장
torch.save(best_model.state_dict(), "model/dnn/best_dnn_model.pth")
print("✅ Best Optuna Model Saved.")

# ==========================================
# 5. 최종 평가 (기존 코드와 동일)
# ==========================================
print("\n--- [4/6] Final Evaluation ---")
best_model.eval()
with torch.no_grad():
    Y_pred_scaled = best_model(X_test_tensor).cpu().numpy()

# 스케일 복원
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)

# 종합 지표
mae = mean_absolute_error(Y_test_values, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test_values, Y_pred))
mape = mean_absolute_percentage_error(Y_test_values, Y_pred)

print(f"  [Overall Performance]")
print(f"  - MAE  : {mae:.4f}")
print(f"  - RMSE : {rmse:.4f}")
print(f"  - MAPE : {mape*100:.2f} (%)")

# 컬럼별 상세 지표
mae_per_col = []
for i, col in enumerate(output_columns):
    c_mae = mean_absolute_error(Y_test_values[:, i], Y_pred[:, i])
    mae_per_col.append(c_mae)

# ==========================================
# 6. 시각화 (기존 코드 유지)
# ==========================================
print("\n--- [5/6] Visualization ---")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. MAE Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=mae_per_col, y=output_columns, hue=output_columns, palette='viridis', legend=False)
plt.title(f'MAE by Output (Optuna Tuned)\nBest Params: {best_params["n_layers"]} Layers, {best_params["optimizer"]}', fontsize=14)
plt.xlabel('MAE (Watt)')
plt.tight_layout()
plt.savefig("eval/dnn/dnn_optuna_mae.png", dpi=300)

# 2. Scatter Plot (Overall)
plt.figure(figsize=(10, 10))
y_true_flat = Y_test_values.flatten()
y_pred_flat = Y_pred.flatten()
plt.scatter(y_true_flat, y_pred_flat, alpha=0.2, s=10, color='darkblue')
min_val = min(y_true_flat.min(), y_pred_flat.min())
max_val = max(y_true_flat.max(), y_pred_flat.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.title(f'Overall True vs Pred (Optuna)\nRMSE: {rmse:.4f}, MAE: {mae:.4f}', fontsize=15)
plt.tight_layout()
plt.savefig("eval/dnn/dnn_optuna_scatter.png", dpi=300)

print("✅ Visualization saved.")

# ==========================================
# 7. CSV 저장
# ==========================================
eval_results = pd.DataFrame()
for i, col in enumerate(output_columns):
    eval_results[f'True_{col}'] = Y_test_values[:, i]
    eval_results[f'Pred_{col}'] = Y_pred[:, i]

eval_results.to_csv("eval/dnn/dnn_optuna_results.csv", index=False)
print("✅ Results CSV saved.")