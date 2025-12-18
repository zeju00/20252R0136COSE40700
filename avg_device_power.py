import os
import re
import glob
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ==============================================================================
# 1. 파싱 헬퍼 함수 (기존과 동일)
# ==============================================================================
# ... (이전 코드와 동일하므로 생략합니다. 꼭 포함시켜 주세요!) ...
def extract_power_from_log(resource_log_path):
    try:
        with open(resource_log_path, 'r', encoding='utf-8') as f: content = f.read()
    except Exception: return None
    power_match = re.search(r"Power \(mW\):\n(.*?)(?:\n\n|\Z)", content, re.DOTALL)
    if not power_match: return None
    timestamps, total_power_values = [], []
    for line in power_match.group(1).strip().split('\n'):
        parts = line.split(':', 1)
        if len(parts) != 2: continue
        try:
            ts = float(parts[0].strip())
            val_dict = ast.literal_eval(parts[1].strip())
            p_val = val_dict.get('tot_power_avg', val_dict.get('tot_power_w', val_dict.get('cpu_power_w', 0) + val_dict.get('gpu_power_w', 0)))
            timestamps.append(ts); total_power_values.append(p_val / 1000.0)
        except: continue
    return pd.DataFrame({'timestamp': timestamps, 'value': total_power_values}) if timestamps else None

def parse_intervals(content, start_label, end_label):
    pattern = re.compile(rf".*?{start_label}\s*=\s*([\d.]+).*?{end_label}\s*=\s*([\d.]+)", re.DOTALL)
    return [(float(s), float(e)) for s, e in pattern.findall(content)]

def get_avg_power(df, start, end):
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    sub = df[mask]
    return sub['value'].mean() if not sub.empty else np.nan

# ==============================================================================
# 2. 메인 분석 로직 (기존과 동일)
# ==============================================================================
def main():
    root_dir = "."; data_records = []
    print(f"🚀 분석 시작: {os.path.abspath(root_dir)}")
    bw_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(d) and not d.startswith('.')]
    for bw_dir in bw_dirs:
        bw_path = os.path.join(root_dir, bw_dir)
        device_dirs = [d for d in os.listdir(bw_path) if os.path.isdir(os.path.join(bw_path, d))]
        for dev_dir in device_dirs:
            lower_name = dev_dir.lower()
            if lower_name.startswith('agx'): device_type = 'AGX Orin'
            elif lower_name.startswith('nx'): device_type = 'Orin NX'
            elif lower_name.startswith('nano'): device_type = 'Orin Nano'
            else: continue
            dev_path = os.path.join(bw_path, dev_dir)
            exp_dirs = glob.glob(os.path.join(dev_path, "learning_sfl-mp_*"))
            for exp_path in exp_dirs:
                try:
                    client_dir = glob.glob(os.path.join(exp_path, "client_*"))[0]
                    res_log = glob.glob(os.path.join(client_dir, "resource_client_*.txt"))[0]
                    time_log = glob.glob(os.path.join(client_dir, "time_client_*.txt"))[0]
                    power_df = extract_power_from_log(res_log)
                    if power_df is None: continue
                    idle_proxy = power_df['value'].quantile(0.05)
                    with open(time_log, 'r', encoding='utf-8') as f: time_content = f.read()
                    fwd_ints = parse_intervals(time_content, "Forward start time", "Forward end time")
                    comm_ints = parse_intervals(time_content, "Smashed send start time", "Smashed send end time")
                    bwd_ints = parse_intervals(time_content, "Backward start time", "Backward end time")
                    n_iter = min(len(fwd_ints), len(comm_ints), len(bwd_ints))
                    for i in range(n_iter):
                        f_avg = get_avg_power(power_df, *fwd_ints[i])
                        c_avg = get_avg_power(power_df, *comm_ints[i])
                        b_avg = get_avg_power(power_df, *bwd_ints[i])
                        if pd.notna(f_avg) and pd.notna(c_avg) and pd.notna(b_avg):
                            data_records.append({'Device': device_type, 'Forward': f_avg, 'Communication': c_avg, 'Backward': b_avg, 'Idle': idle_proxy})
                except: continue

    # ==============================================================================
    # 3. 데이터 시각화 (논문용 스타일 적용)
    # ==============================================================================
    if not data_records: return
    df = pd.DataFrame(data_records)
    df_melted = df.melt(id_vars=['Device', 'Idle'], value_vars=['Forward', 'Communication', 'Backward'], var_name='Phase', value_name='Power')
    df_grouped = df_melted.groupby(['Device', 'Phase'])['Power'].mean().reset_index()
    idle_means = df.groupby('Device')['Idle'].mean()

    # --- [스타일 설정] ---
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "grid.color": "gray", "grid.linestyle": ":"})
    plt.figure(figsize=(8, 5)) # 논문 컬럼에 맞게 가로폭을 줄임 (8인치)

    # 폰트 크기 설정 변수
    FONT_SIZE_TICKS = 14
    FONT_SIZE_LABEL = 16
    FONT_SIZE_LEGEND = 14
    FONT_SIZE_BAR_TEXT = 13
    FONT_SIZE_IDLE_TEXT = 14

    device_order = ['Orin Nano', 'Orin NX', 'AGX Orin']
    existing_order = [d for d in device_order if d in df_grouped['Device'].unique()]

    # (1) 막대 그래프
    ax = sns.barplot(data=df_grouped, x='Device', y='Power', hue='Phase', order=existing_order, palette='viridis', edgecolor='black', linewidth=1.2, alpha=0.9)

    # (2) Idle 선 및 텍스트
    x_coords = range(len(existing_order))
    bar_width = 0.8
    for i, x in enumerate(x_coords):
        dev_name = existing_order[i]
        if dev_name in idle_means:
            idle_val = idle_means[dev_name]
            ax.plot([x-bar_width/2, x+bar_width/2], [idle_val, idle_val], color='red', linestyle='--', linewidth=3, zorder=5)
            ax.text(x, idle_val + 0.8, f"{idle_val:.1f}W", color='red', ha='center', fontweight='bold', fontsize=FONT_SIZE_IDLE_TEXT, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

    # (3) 축, 레이블, 타이틀 설정
    # plt.title('...') # <-- 제목 제거됨
    plt.xlabel('Edge Device', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=10)
    plt.ylabel('Average Power (W)', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=FONT_SIZE_TICKS, fontweight='bold')
    plt.yticks(fontsize=FONT_SIZE_TICKS)

    # (4) 범례 커스텀
    idle_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=3, label='Idle Avg')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(idle_line)
    labels.append('Idle Avg')
    # 범례를 그래프 안쪽 상단으로 이동하여 공간 확보
    plt.legend(handles=handles, labels=labels, title='Phase', title_fontsize=FONT_SIZE_LEGEND, fontsize=FONT_SIZE_LEGEND, loc='upper left', frameon=True, framealpha=0.9)

    # (5) 막대 위 수치 표시
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=4, fontsize=FONT_SIZE_BAR_TEXT, fontweight='bold')

    # (6) 레이아웃 및 저장
    plt.tight_layout(pad=1.5) # 여백 조정
    save_name = 'device_power_paper_ready.png'
    plt.savefig(save_name, dpi=600, bbox_inches='tight') # 고해상도(600dpi) 저장
    print(f"🎉 논문용 그래프 저장 완료: {save_name}")

if __name__ == "__main__":
    main()