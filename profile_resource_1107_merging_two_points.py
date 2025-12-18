import os
import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_value(pattern, text):
    """주어진 패턴으로 텍스트에서 숫자 값을 추출합니다."""
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

def parse_specific_intervals(content, start_label, end_label):
    """주어진 레이블에 대해 (시작, 종료) 타임스탬프 리스트를 추출합니다."""
    intervals = []
    pattern = re.compile(
        rf"{start_label}\s*=\s*([\d.]+)"
        r".*?"
        rf"{end_label}\s*=\s*([\d.]+)",
        re.DOTALL
    )
    matches = pattern.findall(content)
    for start_str, end_str in matches:
        try:
            intervals.append((float(start_str), float(end_str)))
        except ValueError:
            continue
    return intervals


def calculate_average_power_in_intervals(power_df, intervals):
    """전력 데이터프레임과 시간 구간 리스트를 받아, 해당 구간 내의 평균 전력을 계산합니다."""
    if power_df is None or power_df.empty or not intervals:
        return np.nan
    power_in_intervals = pd.concat([
        power_df[(power_df['timestamp'] >= start) & (power_df['timestamp'] <= end)]
        for start, end in intervals
    ])
    return power_in_intervals['value'].mean() if not power_in_intervals.empty else np.nan

def merge_points_pairwise(df):
    """
    데이터프레임의 포인트를 2개씩 묶어서 평균을 계산합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        timestamp와 value 컬럼을 가진 데이터프레임
    
    Returns:
    --------
    pd.DataFrame
        2개씩 묶은 평균값을 가진 데이터프레임
    """
    if df is None or df.empty:
        return df
    
    # 포인트 개수가 홀수인 경우, 마지막 포인트는 그대로 유지
    n_points = len(df)
    if n_points < 2:
        return df
    
    merged_data = []
    for i in range(0, n_points - 1, 2):
        # 2개 포인트의 평균 계산
        avg_timestamp = (df.iloc[i]['timestamp'] + df.iloc[i+1]['timestamp']) / 2
        avg_value = (df.iloc[i]['value'] + df.iloc[i+1]['value']) / 2
        merged_data.append({'timestamp': avg_timestamp, 'value': avg_value})
    
    # 홀수 개인 경우 마지막 포인트 추가
    if n_points % 2 == 1:
        merged_data.append({'timestamp': df.iloc[-1]['timestamp'], 'value': df.iloc[-1]['value']})
    
    return pd.DataFrame(merged_data)

def extract_all_timeseries(resource_log_path, dir_name):
    """리소스 로그 파일을 파싱하여 데이터프레임 딕셔너리로 반환합니다."""
    try:
        with open(resource_log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return {}

    all_ts_data = {}
    simple_metrics = {
        'cpu_usage': r"Process CPU Usage \(%\):\n(.*?)(?:\n\n|\Z)",
        'cpu_temp': r"CPU Temperature \(.C\):\n(.*?)(?:\n\n|\Z)",
        'gpu_temp': r"GPU Temperature \(.C\):\n(.*?)(?:\n\n|\Z)",
        'gpu_load': r"GPU Load \(%\):\n(.*?)(?:\n\n|\Z)",
        'active_cpu_cores': r"Active CPU Cores \(Count\):\n(.*?)(?:\n\n|\Z)",
    }
    for name, pattern in simple_metrics.items():
        match = re.search(pattern, content, re.DOTALL)
        if not match: continue
        lines = match.group(1).strip().split('\n')
        timestamps, values = [], []
        for line in lines:
            parts = line.split(':', 1)
            if len(parts) != 2: continue
            timestamp_str, val_part = parts[0].strip(), parts[1].strip()
            if val_part.startswith('{') and val_part.endswith('}'):
                try:
                    data_dict = ast.literal_eval(val_part)
                    final_value = None
                    if name == 'cpu_usage': final_value = sum(data_dict.values())
                    elif name == 'cpu_temp': final_value = np.mean(list(data_dict.values()))
                    if final_value is not None:
                        timestamps.append(float(timestamp_str))
                        values.append(final_value)
                except (ValueError, SyntaxError, TypeError): continue
            else:
                num_match = re.search(r'([\d.]+)', val_part)
                if num_match:
                    try:
                        timestamps.append(float(timestamp_str))
                        values.append(float(num_match.group(1)))
                    except ValueError: continue
        if timestamps:
            df = pd.DataFrame({'timestamp': timestamps, 'value': values})
            all_ts_data[name] = df

    power_match = re.search(r"Power \(mW\):\n(.*?)(?:\n\n|\Z)", content, re.DOTALL)
    if power_match:
        power_data = {}
        for line in power_match.group(1).strip().split('\n'):
            parts = line.split(':', 1)
            if len(parts) != 2: continue
            timestamp_str, val_part = parts[0].strip(), parts[1].strip()
            if val_part.startswith('{') and val_part.endswith('}'):
                try:
                    power_dict = ast.literal_eval(val_part)
                    for key, value in power_dict.items():
                        if not key.lower().endswith('_avg'):
                            df_name = f"{key.lower()}_w"
                            if df_name not in power_data:
                                power_data[df_name] = {'timestamps': [], 'values': []}
                            power_data[df_name]['timestamps'].append(float(timestamp_str))
                            power_data[df_name]['values'].append(value / 1000.0)
                except (ValueError, SyntaxError): continue
        for name, data in power_data.items():
            if data['timestamps']:
                df = pd.DataFrame({'timestamp': data['timestamps'], 'value': data['values']})
                all_ts_data[name] = df
    return all_ts_data

def plot_individual_graphs(timeseries_data, output_dir, exp_name, prefix, intervals=None, global_start_time=None, epoch_duration=None, avg_max_power=None, avg_min_power=None):
    """하나의 실험에 대한 개별 리소스 메트릭 그래프를 생성하고 저장합니다."""
    metric_labels = {
        'cpu_usage': "Usage (%)", 'cpu_temp': "Temperature (°C)",
        'gpu_temp': "Temperature (°C)", 'gpu_load': "Load (%)",
        'active_cpu_cores': "Active Cores (Count)",
        'cpu_gpu_power_w': "Power (W)", 'soc_power_w': "Power (W)",
        'cpu_power_w': "Power (W)", 'gpu_power_w': "Power (W)",
        'tot_power_w': "Total Power (W)",
        'summed_power_w': "Power (W)"
    }
    print(f"  - Generating graphs for {prefix} in {os.path.basename(output_dir)}...")
    for name, df in timeseries_data.items():
        if df.empty: continue
        start_time = global_start_time if global_start_time is not None else df['timestamp'].iloc[0]
        time_sec_relative = df['timestamp'] - start_time
        plt.figure(figsize=(12, 6))
        plt.plot(time_sec_relative, df['value'], marker='o', linestyle='-', markersize=3, zorder=2)
        title_name = "Summed Power" if name == 'summed_power_w' else "Total Power" if name == 'tot_power_w' else name.replace("_w", "").replace("_", " ").title()
        plt.title(f'[{exp_name}] {prefix.title()} {title_name} Over Time')
        plt.xlabel("Time (seconds)")
        plt.ylabel(metric_labels.get(name, 'Value'))
        plt.grid(True, zorder=0)
        
        if intervals:
            for start, end in intervals.get('computation', []):
                plt.axvspan(start - start_time, end - start_time, color='lightblue', alpha=0.3, label='Computation', zorder=1)
            for start, end in intervals.get('communication', []):
                plt.axvspan(start - start_time, end - start_time, color='lightgreen', alpha=0.4, label='Communication', zorder=1)
            for start, end in intervals.get('forward', []):
                plt.axvspan(start - start_time, end - start_time, color='sandybrown', alpha=0.5, label='Forward', zorder=1)
            for start, end in intervals.get('backward', []):
                plt.axvspan(start - start_time, end - start_time, color='lightcoral', alpha=0.5, label='Backward', zorder=1)
            
            if name == 'summed_power_w':
                if avg_max_power is not None:
                    plt.axhline(y=avg_max_power, color='red', linestyle='--', label=f'Avg Max Power: {avg_max_power:.2f}W')
                if avg_min_power is not None:
                    plt.axhline(y=avg_min_power, color='blue', linestyle='--', label=f'Avg Min Power: {avg_min_power:.2f}W')
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label: plt.legend(by_label.values(), by_label.keys())
        
        if epoch_duration is not None:
            plt.xlim(0, epoch_duration)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_{name}.png'), dpi=150)
        plt.close()

def plot_graphs_per_epoch(exp_path, exp_name, epoch_intervals, all_data_for_epochs, plot_intervals, iteration_stats_dict):
    """각 Epoch별로 통합 그래프를 생성하고 저장합니다."""
    for epoch_index, (epoch_start, epoch_end) in enumerate(epoch_intervals, start=1):
        print(f"  - Generating graphs for Epoch {epoch_index}...")
        epoch_output_dir = os.path.join(exp_path, f"epoch_{epoch_index}_plots")
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        epoch_duration = epoch_end - epoch_start
        
        for prefix, timeseries_data in all_data_for_epochs.items():
            if not timeseries_data: continue
            epoch_data = {name: df[(df['timestamp'] >= epoch_start) & (df['timestamp'] <= epoch_end)].copy() 
                         for name, df in timeseries_data.items()}
            if not any(not df.empty for df in epoch_data.values()): continue
            plot_individual_graphs(epoch_data, epoch_output_dir, exp_name, prefix, plot_intervals, epoch_start, epoch_duration)

def generate_epoch_reports(exp_path, epoch_intervals, all_data_for_epochs, computation_intervals, communication_intervals, iteration_stats_dict):
    """각 에폭에 대한 요약 리포트를 생성합니다."""
    if 'client' not in all_data_for_epochs or not all_data_for_epochs['client']: return
    client_data = all_data_for_epochs['client']
    power_key = 'summed_power_w' if 'summed_power_w' in client_data else None
    if not power_key or power_key not in client_data or client_data[power_key].empty: return
    power_df = client_data[power_key]
    
    epoch_stats = []
    for epoch_index, (epoch_start, epoch_end) in enumerate(epoch_intervals, start=1):
        epoch_data = power_df[(power_df['timestamp'] >= epoch_start) & (power_df['timestamp'] <= epoch_end)]
        if epoch_data.empty: continue
        avg_power_epoch = epoch_data['value'].mean()
        max_power_epoch = epoch_data['value'].max()
        min_power_epoch = epoch_data['value'].min()
        
        epoch_computation = [(s, e) for s, e in computation_intervals if s >= epoch_start and e <= epoch_end]
        epoch_communication = [(s, e) for s, e in communication_intervals if s >= epoch_start and e <= epoch_end]
        avg_comp_power = calculate_average_power_in_intervals(power_df, epoch_computation)
        avg_comm_power = calculate_average_power_in_intervals(power_df, epoch_communication)
        
        epoch_stats.append({
            'epoch': epoch_index,
            'avg_summed_power_w': avg_power_epoch,
            'max_summed_power_w': max_power_epoch,
            'min_summed_power_w': min_power_epoch,
            'avg_comp_power_w': avg_comp_power,
            'avg_comm_power_w': avg_comm_power
        })
    if epoch_stats:
        df_epoch_stats = pd.DataFrame(epoch_stats)
        epoch_report_path = os.path.join(exp_path, 'epoch_summary_report.csv')
        df_epoch_stats.to_csv(epoch_report_path, index=False)
        print(f"  -> Epoch summary report saved to: {epoch_report_path}")

def compute_slope(df, start_time, end_time):
    """주어진 시간 구간 내에서 전력 데이터의 기울기(slope)를 계산합니다."""
    if df is None or df.empty: return np.nan
    segment = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    if len(segment) < 2: return np.nan
    segment.loc[:, 'rel_time'] = segment['timestamp'] - segment['timestamp'].iloc[0]
    from scipy.stats import linregress
    res = linregress(segment['rel_time'], segment['value'])
    return res.slope

def compute_integral(df, start_time, end_time):
    """주어진 시간 구간 내에서 전력 데이터의 적분(integral)을 계산합니다."""
    if df is None or df.empty: return np.nan
    segment = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    if segment.empty: return np.nan
    from scipy.integrate import trapz
    return trapz(segment['value'], segment['timestamp'])

def create_aggregate_plots(all_timeseries_collection, dir_path):
    """모든 split_index에 대한 통합 플롯을 생성합니다."""
    if not all_timeseries_collection: return
    dir_name = os.path.basename(dir_path)
    aggregated_data = {}
    for entry in all_timeseries_collection:
        split_index = entry['split_index']
        ts_data = entry['data']
        start_time = entry['start_time']
        for key, df in ts_data.items():
            if df.empty: continue
            if key not in aggregated_data: aggregated_data[key] = []
            df_shifted = df.copy()
            df_shifted['timestamp'] = df_shifted['timestamp'] - start_time
            aggregated_data[key].append((split_index, df_shifted))
    
    metric_labels = {
        'client_cpu_usage': "Client CPU Usage (%)", 'client_cpu_temp': "Client CPU Temperature (°C)",
        'client_gpu_temp': "Client GPU Temperature (°C)", 'client_gpu_load': "Client GPU Load (%)",
        'client_active_cpu_cores': "Client Active CPU Cores (Count)",
        'client_cpu_gpu_power_w': "Client CPU+GPU Power (W)", 'client_soc_power_w': "Client SoC Power (W)",
        'client_cpu_power_w': "Client CPU Power (W)", 'client_gpu_power_w': "Client GPU Power (W)",
        'client_tot_power_w': "Client Total Power (W)", 'client_summed_power_w': "Client Summed Power (W)",
        'runner_cpu_usage': "Runner CPU Usage (%)", 'runner_cpu_temp': "Runner CPU Temperature (°C)",
        'runner_gpu_temp': "Runner GPU Temperature (°C)", 'runner_gpu_load': "Runner GPU Load (%)",
        'runner_active_cpu_cores': "Runner Active CPU Cores (Count)",
        'runner_cpu_gpu_power_w': "Runner CPU+GPU Power (W)", 'runner_soc_power_w': "Runner SoC Power (W)",
        'runner_cpu_power_w': "Runner CPU Power (W)", 'runner_gpu_power_w': "Runner GPU Power (W)",
        'runner_tot_power_w': "Runner Total Power (W)", 'runner_summed_power_w': "Runner Summed Power (W)"
    }
    
    print(f"\n- Creating aggregate plots for {dir_name}...")
    for key, split_data_list in aggregated_data.items():
        plt.figure(figsize=(14, 8))
        for split_index, df in sorted(split_data_list, key=lambda x: x[0]):
            plt.plot(df['timestamp'], df['value'], marker='o', linestyle='-', markersize=3, label=f'Split {split_index}')
        title_base = key.replace("_w", "").replace("_", " ").title()
        if key.endswith("_summed_power_w"): title_base = key.replace("_summed_power_w", " Summed Power")
        elif key.endswith("_tot_power_w"): title_base = key.replace("_tot_power_w", " Total Power")
        plt.title(f'[{dir_name}] Aggregate {title_base} Over Time (All Split Indices)')
        plt.xlabel("Time (seconds, relative to epoch start)")
        plt.ylabel(metric_labels.get(key, 'Value'))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, f'aggregate_{key}.png'), dpi=150)
        plt.close()
    print(f"  -> Aggregate plots saved to: {dir_path}")

def plot_single_representative_iteration(rep_iter_data, exp_path, exp_name):
    """
    단일 대표 이터레이션(forward, comm, backward)의 summed_power_w를 
    상대 시간(0부터 시작)으로 변환하여 플롯합니다.
    """
    if not rep_iter_data or 'df' not in rep_iter_data or 'intervals' not in rep_iter_data:
        return
    
    df = rep_iter_data['df']
    intervals = rep_iter_data['intervals']
    split_index = rep_iter_data.get('split_index', 'unknown')
    
    if df is None or df.empty:
        return
    
    # 전체 이터레이션의 시작 시간을 0으로 설정
    iter_start = df['timestamp'].iloc[0]
    df_shifted = df.copy()
    df_shifted['timestamp'] = df_shifted['timestamp'] - iter_start
    
    plt.figure(figsize=(14, 6))
    plt.plot(df_shifted['timestamp'], df_shifted['value'], marker='o', linestyle='-', markersize=4, color='black', zorder=3, label='Summed Power')
    
    # Forward 구간 표시
    if 'forward' in intervals:
        for start, end in intervals['forward']:
            plt.axvspan(start, end, color='sandybrown', alpha=0.5, label='Forward' if 'Forward' not in plt.gca().get_legend_handles_labels()[1] else "", zorder=1)
    
    # Communication 구간 표시
    if 'communication' in intervals:
        for start, end in intervals['communication']:
            plt.axvspan(start, end, color='lightgreen', alpha=0.4, label='Communication' if 'Communication' not in plt.gca().get_legend_handles_labels()[1] else "", zorder=1)
    
    # Backward 구간 표시
    if 'backward' in intervals:
        for start, end in intervals['backward']:
            plt.axvspan(start, end, color='lightcoral', alpha=0.5, label='Backward' if 'Backward' not in plt.gca().get_legend_handles_labels()[1] else "", zorder=1)
    
    plt.title(f'[{exp_name}] Representative Iteration - Summed Power (Split {split_index})')
    plt.xlabel("Relative Time (seconds)")
    plt.ylabel("Summed Power (W)")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.tight_layout()
    
    output_path = os.path.join(exp_path, f'representative_iteration_split_{split_index}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  -> Representative iteration plot saved to: {output_path}")

def plot_representative_iteration_timeline(all_representative_iterations, dir_path, dir_name):
    """
    모든 split_index에 대한 대표 이터레이션을 하나의 타임라인 그래프로 시각화합니다.
    각 split의 forward, comm, backward를 순차적으로 배치합니다.
    """
    if not all_representative_iterations:
        return
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    current_time = 0
    all_segments = []
    
    # 각 split_index에 대해 처리
    for rep_data in sorted(all_representative_iterations, key=lambda x: x.get('split_index', 0)):
        split_index = rep_data.get('split_index', 'unknown')
        df = rep_data.get('df')
        intervals = rep_data.get('intervals')
        
        if df is None or df.empty or not intervals:
            continue
        
        # Forward 구간
        if 'forward' in intervals and intervals['forward']:
            fwd_start_rel, fwd_end_rel = intervals['forward'][0]
            fwd_df = df[(df['timestamp'] >= (df['timestamp'].iloc[0] + fwd_start_rel)) & 
                       (df['timestamp'] <= (df['timestamp'].iloc[0] + fwd_end_rel))].copy()
            if not fwd_df.empty:
                fwd_df['timeline_time'] = current_time + (fwd_df['timestamp'] - fwd_df['timestamp'].iloc[0])
                all_segments.append({
                    'split': split_index,
                    'phase': 'forward',
                    'df': fwd_df,
                    'start': current_time,
                    'end': current_time + (fwd_end_rel - fwd_start_rel),
                    'color': 'sandybrown'
                })
                current_time += (fwd_end_rel - fwd_start_rel)
        
        # Communication 구간
        if 'communication' in intervals and intervals['communication']:
            comm_start_rel, comm_end_rel = intervals['communication'][0]
            comm_df = df[(df['timestamp'] >= (df['timestamp'].iloc[0] + comm_start_rel)) & 
                        (df['timestamp'] <= (df['timestamp'].iloc[0] + comm_end_rel))].copy()
            if not comm_df.empty:
                comm_df['timeline_time'] = current_time + (comm_df['timestamp'] - comm_df['timestamp'].iloc[0])
                all_segments.append({
                    'split': split_index,
                    'phase': 'communication',
                    'df': comm_df,
                    'start': current_time,
                    'end': current_time + (comm_end_rel - comm_start_rel),
                    'color': 'lightgreen'
                })
                current_time += (comm_end_rel - comm_start_rel)
        
        # Backward 구간
        if 'backward' in intervals and intervals['backward']:
            bwd_start_rel, bwd_end_rel = intervals['backward'][0]
            bwd_df = df[(df['timestamp'] >= (df['timestamp'].iloc[0] + bwd_start_rel)) & 
                       (df['timestamp'] <= (df['timestamp'].iloc[0] + bwd_end_rel))].copy()
            if not bwd_df.empty:
                bwd_df['timeline_time'] = current_time + (bwd_df['timestamp'] - bwd_df['timestamp'].iloc[0])
                all_segments.append({
                    'split': split_index,
                    'phase': 'backward',
                    'df': bwd_df,
                    'start': current_time,
                    'end': current_time + (bwd_end_rel - bwd_start_rel),
                    'color': 'lightcoral'
                })
                current_time += (bwd_end_rel - bwd_start_rel)
    
    # 각 구간을 그래프에 그리기
    phase_labels_added = set()
    for segment in all_segments:
        df_seg = segment['df']
        phase = segment['phase']
        color = segment['color']
        
        # 배경색 표시
        ax.axvspan(segment['start'], segment['end'], color=color, alpha=0.3, zorder=1)
        
        # 전력 데이터 플롯
        label = f"{phase.capitalize()}" if phase not in phase_labels_added else ""
        ax.plot(df_seg['timeline_time'], df_seg['value'], marker='o', linestyle='-', 
               markersize=3, color='black', zorder=2, label=label)
        phase_labels_added.add(phase)
        
        # Split 구분선
        if segment == all_segments[-1] or all_segments[all_segments.index(segment) + 1]['split'] != segment['split']:
            ax.axvline(x=segment['end'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
            ax.text(segment['end'], ax.get_ylim()[1] * 0.95, f'Split {segment["split"]}', 
                   rotation=90, verticalalignment='top', fontsize=9, color='red')
    
    ax.set_xlabel("Timeline (seconds)")
    ax.set_ylabel("Summed Power (W)")
    ax.set_title(f'[{dir_name}] Representative Iteration Timeline - All Splits')
    ax.legend(loc='upper right')
    ax.grid(True, zorder=0)
    plt.tight_layout()
    
    output_path = os.path.join(dir_path, 'representative_iteration_timeline.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  -> Representative iteration timeline saved to: {output_path}")

def analyze_experiment(exp_path, dir_name):
    """하나의 실험 디렉토리를 분석하여 결과를 반환합니다."""
    exp_name = os.path.basename(exp_path)
    match = re.search(r'learning_sfl-mp_(\d+)_', exp_name)
    if not match: return None, None, None, None
    split_index = int(match.group(1))
    
    client_resource_log = os.path.join(exp_path, 'client', 'resource_log.txt')
    runner_resource_log = os.path.join(exp_path, 'runner', 'resource_log.txt')
    client_log = os.path.join(exp_path, 'client', 'main.log')
    runner_log = os.path.join(exp_path, 'runner', 'main.log')
    
    # 디렉토리 이름 확인: nx 또는 nano로 시작하는지 체크
    should_merge_points = dir_name.startswith('nx') or dir_name.startswith('nano')
    
    client_ts_data = extract_all_timeseries(client_resource_log, dir_name) if os.path.exists(client_resource_log) else {}
    runner_ts_data = extract_all_timeseries(runner_resource_log, dir_name) if os.path.exists(runner_resource_log) else {}
    
    combined_results = {'split_index': split_index}
    
    for prefix, ts_data, resource_log in [('client', client_ts_data, client_resource_log), ('runner', runner_ts_data, runner_resource_log)]:
        if not os.path.exists(resource_log): continue
        with open(resource_log, 'r', encoding='utf-8') as f: content = f.read()
        
        resource_cols = ['cpu_usage', 'cpu_temp', 'gpu_temp', 'gpu_load', 'active_cpu_cores']
        power_cols = ['tot_power_w', 'cpu_gpu_power_w', 'soc_power_w', 'cpu_power_w', 'gpu_power_w']
        for col in resource_cols + power_cols:
            if col in ts_data and not ts_data[col].empty:
                combined_results[f'{prefix}_avg_{col}'] = ts_data[col]['value'].mean()
                combined_results[f'{prefix}_max_{col}'] = ts_data[col]['value'].max()
        
        if 'cpu_gpu_power_w' in ts_data and not ts_data['cpu_gpu_power_w'].empty:
            soc_df = ts_data.get('soc_power_w')
            if soc_df is not None and not soc_df.empty:
                merged_df = pd.merge(ts_data['cpu_gpu_power_w'], soc_df, on='timestamp', suffixes=('_cpugpu', '_soc'))
                merged_df['summed'] = merged_df['value_cpugpu'] + merged_df['value_soc']
                summed_power_df = merged_df[['timestamp', 'summed']].rename(columns={'summed': 'value'})
                ts_data['summed_power_w'] = summed_power_df
                combined_results[f'{prefix}_avg_summed_power_w'] = summed_power_df['value'].mean()
                combined_results[f'{prefix}_max_summed_power_w'] = summed_power_df['value'].max()
    
    computation_intervals, communication_intervals, forward_intervals, backward_intervals = [], [], [], []
    epoch_intervals, iteration_stats_dict = [], {}
    epoch1_start_time = None
    representative_iteration_data = None
    
    if os.path.exists(client_log):
        with open(client_log, 'r', encoding='utf-8') as f: log_content = f.read()
        
        combined_results['avg_iteration_duration'] = parse_value(r"Average Iteration Time:\s*([\d.]+)", log_content)
        combined_results['avg_tokens_per_second'] = parse_value(r"Tokens per Second:\s*([\d.]+)", log_content)
        combined_results['total_training_duration'] = parse_value(r"Total Training Time:\s*([\d.]+)", log_content)
        
        computation_intervals = parse_specific_intervals(log_content, "computation_start", "computation_end")
        communication_intervals = parse_specific_intervals(log_content, "communication_start", "communication_end")
        forward_intervals = parse_specific_intervals(log_content, "forward_start", "forward_end")
        backward_intervals = parse_specific_intervals(log_content, "backward_start", "backward_end")
        
        epoch_pattern = re.compile(r"Epoch (\d+) started.*?epoch_start\s*=\s*([\d.]+).*?epoch_end\s*=\s*([\d.]+)", re.DOTALL)
        epoch_matches = epoch_pattern.findall(log_content)
        for epoch_num_str, start_str, end_str in epoch_matches:
            try:
                epoch_num = int(epoch_num_str)
                start_time = float(start_str)
                end_time = float(end_str)
                epoch_intervals.append((start_time, end_time))
                if epoch_num == 1: epoch1_start_time = start_time
            except ValueError: continue
        
        iter_pattern = re.compile(
            r"Iteration (\d+)/\d+ \(Epoch (\d+)\).*?"
            r"forward_start\s*=\s*([\d.]+).*?"
            r"forward_end\s*=\s*([\d.]+).*?"
            r"communication_start\s*=\s*([\d.]+).*?"
            r"communication_end\s*=\s*([\d.]+).*?"
            r"backward_start\s*=\s*([\d.]+).*?"
            r"backward_end\s*=\s*([\d.]+)",
            re.DOTALL
        )
        iter_matches = iter_pattern.findall(log_content)
        
        for match in iter_matches:
            try:
                iter_num = int(match[0])
                epoch_num = int(match[1])
                fwd_start = float(match[2])
                fwd_end = float(match[3])
                comm_start = float(match[4])
                comm_end = float(match[5])
                bwd_start = float(match[6])
                bwd_end = float(match[7])
                
                if epoch_num not in iteration_stats_dict:
                    iteration_stats_dict[epoch_num] = []
                
                iteration_stats_dict[epoch_num].append({
                    'iteration': iter_num,
                    'forward_start': fwd_start,
                    'forward_end': fwd_end,
                    'communication_start': comm_start,
                    'communication_end': comm_end,
                    'backward_start': bwd_start,
                    'backward_end': bwd_end
                })
            except (ValueError, IndexError):
                continue
        
        if forward_intervals: combined_results['avg_forward_duration'] = np.mean([end - start for start, end in forward_intervals])
        if backward_intervals: combined_results['avg_backward_duration'] = np.mean([end - start for start, end in backward_intervals])
        if communication_intervals: combined_results['avg_comm_duration'] = np.mean([end - start for start, end in communication_intervals])
        
        client_power_df = client_ts_data.get('summed_power_w')
        if client_power_df is not None and not client_power_df.empty:
            if communication_intervals:
                combined_results['client_comm_avg_gpu_load'] = calculate_average_power_in_intervals(client_ts_data.get('gpu_load'), communication_intervals)
                combined_results['client_comm_avg_gpu_power_w'] = calculate_average_power_in_intervals(client_ts_data.get('gpu_power_w'), communication_intervals)
                combined_results['client_comm_avg_cpu_gpu_power_w'] = calculate_average_power_in_intervals(client_ts_data.get('cpu_gpu_power_w'), communication_intervals)
                combined_results['client_comm_avg_summed_power_w'] = calculate_average_power_in_intervals(client_power_df, communication_intervals)
            if computation_intervals:
                combined_results['client_comp_avg_gpu_load'] = calculate_average_power_in_intervals(client_ts_data.get('gpu_load'), computation_intervals)
                combined_results['client_comp_avg_gpu_power_w'] = calculate_average_power_in_intervals(client_ts_data.get('gpu_power_w'), computation_intervals)
                combined_results['client_comp_avg_cpu_gpu_power_w'] = calculate_average_power_in_intervals(client_ts_data.get('cpu_gpu_power_w'), computation_intervals)
                combined_results['client_comp_avg_summed_power_w'] = calculate_average_power_in_intervals(client_power_df, computation_intervals)
            if forward_intervals: combined_results['client_forward_avg_summed_power_w'] = calculate_average_power_in_intervals(client_power_df, forward_intervals)
            if backward_intervals: combined_results['client_backward_avg_summed_power_w'] = calculate_average_power_in_intervals(client_power_df, backward_intervals)
            
            combined_results['client_avg_forward_summed_power_slope'] = np.mean([compute_slope(client_power_df, s, e) for s, e in forward_intervals]) if forward_intervals else np.nan
            combined_results['client_avg_backward_summed_power_slope'] = np.mean([compute_slope(client_power_df, s, e) for s, e in backward_intervals]) if backward_intervals else np.nan
            combined_results['client_avg_comm_summed_power_slope'] = np.mean([compute_slope(client_power_df, s, e) for s, e in communication_intervals]) if communication_intervals else np.nan
            
            combined_results['client_avg_forward_summed_power_integral'] = np.mean([compute_integral(client_power_df, s, e) for s, e in forward_intervals]) if forward_intervals else np.nan
            combined_results['client_avg_backward_summed_power_integral'] = np.mean([compute_integral(client_power_df, s, e) for s, e in backward_intervals]) if backward_intervals else np.nan
            combined_results['client_avg_comm_summed_power_integral'] = np.mean([compute_integral(client_power_df, s, e) for s, e in communication_intervals]) if communication_intervals else np.nan
            
            # 대표 이터레이션 선택 및 플롯 (첫 번째 에폭의 중간 이터레이션)
            if iteration_stats_dict and client_power_df is not None and not client_power_df.empty:
                first_epoch_num = min(iteration_stats_dict.keys())
                iterations = iteration_stats_dict[first_epoch_num]
                
                if iterations:
                    mid_index = len(iterations) // 2
                    rep_iter = iterations[mid_index]
                    
                    rep_iter_start = rep_iter['forward_start']
                    rep_iter_end = rep_iter['backward_end']
                    
                    rep_power_df = client_power_df[
                        (client_power_df['timestamp'] >= rep_iter_start) &
                        (client_power_df['timestamp'] <= rep_iter_end)
                    ].copy()
                    
                    if not rep_power_df.empty:
                        # nx 또는 nano로 시작하는 경우 포인트 병합
                        if should_merge_points:
                            print(f"  -> Merging points for {dir_name} (split {split_index})")
                            rep_power_df = merge_points_pairwise(rep_power_df)
                        
                        # 각 구간별 포인트 개수 카운트
                        rep_fwd_interval = (rep_iter['forward_start'], rep_iter['forward_end'])
                        rep_comm_interval = (rep_iter['communication_start'], rep_iter['communication_end'])
                        rep_bwd_interval = (rep_iter['backward_start'], rep_iter['backward_end'])
                        
                        # 병합 후의 포인트 개수 카운트
                        fwd_points_count = len(rep_power_df[
                            (rep_power_df['timestamp'] >= rep_fwd_interval[0]) &
                            (rep_power_df['timestamp'] <= rep_fwd_interval[1])
                        ])
                        comm_points_count = len(rep_power_df[
                            (rep_power_df['timestamp'] >= rep_comm_interval[0]) &
                            (rep_power_df['timestamp'] <= rep_comm_interval[1])
                        ])
                        bwd_points_count = len(rep_power_df[
                            (rep_power_df['timestamp'] >= rep_bwd_interval[0]) &
                            (rep_power_df['timestamp'] <= rep_bwd_interval[1])
                        ])
                        
                        combined_results['client_rep_iter_fwd_points'] = fwd_points_count
                        combined_results['client_rep_iter_comm_points'] = comm_points_count
                        combined_results['client_rep_iter_bwd_points'] = bwd_points_count
                        
                        rep_intervals = {
                            'forward': [(rep_fwd_interval[0] - rep_iter_start, rep_fwd_interval[1] - rep_iter_start)],
                            'communication': [(rep_comm_interval[0] - rep_iter_start, rep_comm_interval[1] - rep_iter_start)],
                            'backward': [(rep_bwd_interval[0] - rep_iter_start, rep_bwd_interval[1] - rep_iter_start)]
                        }

                        representative_iteration_data = {
                            'split_index': split_index,
                            'df': rep_power_df,
                            'intervals': rep_intervals
                        }
                        
                        if representative_iteration_data:
                            plot_single_representative_iteration(representative_iteration_data, exp_path, exp_name)

    total_communication_duration = sum(end - start for start, end in communication_intervals)
    total_training_duration = combined_results.get('total_training_duration')

    if total_training_duration is not None and total_training_duration > 0:
        total_communication_duration = min(total_communication_duration, total_training_duration)
        total_computation_duration = total_training_duration - total_communication_duration
        communication_time_percentage = (total_communication_duration / total_training_duration) * 100
        computation_time_percentage = (total_computation_duration / total_training_duration) * 100
    else:
        total_computation_duration, communication_time_percentage, computation_time_percentage = None, None, None

    combined_results.update({
        'total_communication_duration': total_communication_duration,
        'total_computation_duration': total_computation_duration,
        'communication_time_percentage': communication_time_percentage,
        'computation_time_percentage': computation_time_percentage
    })
    
    global_start_time = epoch1_start_time if epoch1_start_time else 0

    plot_intervals = {
        'computation': computation_intervals, 
        'communication': communication_intervals,
        'forward': forward_intervals,
        'backward': backward_intervals
    }

    if client_ts_data: plot_individual_graphs(client_ts_data, exp_path, exp_name, "client", plot_intervals, global_start_time)
    if runner_ts_data: plot_individual_graphs(runner_ts_data, exp_path, exp_name, "runner", plot_intervals, global_start_time)

    if epoch_intervals:
        all_data_for_epochs = {"client": client_ts_data, "runner": runner_ts_data}
        plot_graphs_per_epoch(exp_path, exp_name, epoch_intervals, all_data_for_epochs, plot_intervals, iteration_stats_dict)
        generate_epoch_reports(exp_path, epoch_intervals, all_data_for_epochs, computation_intervals, communication_intervals, iteration_stats_dict)

    return combined_results, {**{f"client_{k}": v for k, v in client_ts_data.items()}, **{f"runner_{k}": v for k, v in runner_ts_data.items()}}, global_start_time, representative_iteration_data

def process_bandwidth_dir(dir_path, top_level_dir):
    dir_name = os.path.basename(dir_path)
    print(f"\n{'='*30} Processing Directory: {dir_name} {'='*30}")
    
    all_results, all_timeseries_collection = [], []
    all_representative_iterations = []

    for exp_name in sorted([d for d in os.listdir(dir_path) if d.startswith("learning_sfl-mp_")]):
        exp_path = os.path.join(dir_path, exp_name)
        if os.path.isdir(exp_path):
            print(f"- Analyzing {exp_name}...")
            exp_data, timeseries_data, start_time, rep_data = analyze_experiment(exp_path, dir_name)
            if exp_data: all_results.append(exp_data)
            if timeseries_data:
                all_timeseries_collection.append({'split_index': exp_data['split_index'], 'data': timeseries_data, 'start_time': start_time})
            if rep_data:
                all_representative_iterations.append(rep_data)

    if not all_results: return

    df = pd.DataFrame(all_results).sort_values(by='split_index').reset_index(drop=True)
    epsilon = 1e-9
    for prefix in ['client', 'runner']:
        if f'{prefix}_avg_tot_power_w' in df.columns:
            valid = df[f'{prefix}_avg_tot_power_w'].notna()
            df.loc[valid, f'{prefix}_energy_efficiency'] = (1 / (df.loc[valid, f'{prefix}_avg_tot_power_w'] + epsilon)) * (1 / (df.loc[valid, 'avg_iteration_duration'] + epsilon))

    base_cols = [
        'split_index', 'avg_iteration_duration', 'avg_tokens_per_second',
        'total_training_duration', 'total_computation_duration',
        'total_communication_duration', 'computation_time_percentage',
        'communication_time_percentage', 'avg_forward_duration', 
        'avg_backward_duration', 'avg_comm_duration'
    ]
    
    resource_cols = ['cpu_usage', 'cpu_temp', 'gpu_temp', 'gpu_load', 'active_cpu_cores']
    power_cols = ['tot_power_w', 'cpu_gpu_power_w', 'soc_power_w', 'cpu_power_w', 'gpu_power_w', 'summed_power_w']
    
    analysis_cols = [
        'comm_avg_gpu_load', 'comm_avg_gpu_power_w', 'comm_avg_cpu_gpu_power_w', 'comm_avg_summed_power_w',
        'comp_avg_gpu_load', 'comp_avg_gpu_power_w', 'comp_avg_cpu_gpu_power_w', 'comp_avg_summed_power_w',
        'forward_avg_summed_power_w', 'backward_avg_summed_power_w',
        'avg_forward_summed_power_slope', 'avg_backward_summed_power_slope', 'avg_comm_summed_power_slope',
        'avg_forward_summed_power_integral', 'avg_backward_summed_power_integral', 'avg_comm_summed_power_integral'
    ]
    
    rep_iter_cols = [
        'rep_iter_fwd_points',
        'rep_iter_comm_points',
        'rep_iter_bwd_points'
    ]
    
    client_cols = ['client_energy_efficiency'] + \
                  [f'client_{stat}_{res}' for res in resource_cols + power_cols for stat in ['avg', 'max']] + \
                  [f'client_{col}' for col in analysis_cols] + \
                  [f'client_{col}' for col in rep_iter_cols]
    
    df = df.reindex(columns=base_cols + client_cols)
    
    output_csv = os.path.join(top_level_dir, f'experiment_results_{dir_name}.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n-> Analysis for {dir_name} saved to: {output_csv}\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(df)
    
    print(f"\n- Creating simple power summary for {dir_name}...")
    
    power_summary_cols = [
        'split_index',
        'client_forward_avg_summed_power_w',
        'client_comm_avg_summed_power_w',
        'client_backward_avg_summed_power_w',
        'runner_forward_avg_summed_power_w',
        'runner_comm_avg_summed_power_w',
        'runner_backward_avg_summed_power_w',
        
        'client_rep_iter_fwd_points',
        'client_rep_iter_comm_points',
        'client_rep_iter_bwd_points'
    ]
    
    existing_power_cols = ['split_index'] + [col for col in power_summary_cols[1:] if col in df.columns]
    power_summary_df = df[existing_power_cols]
    
    power_summary_output_csv = os.path.join(top_level_dir, f'power_summary_{dir_name}.csv')
    power_summary_df.to_csv(power_summary_output_csv, index=False)
    
    print(f"-> Simple power summary saved to: {power_summary_output_csv}\n")
    
    if all_timeseries_collection:
        create_aggregate_plots(all_timeseries_collection, dir_path)
    
    if all_representative_iterations:
        plot_representative_iteration_timeline(all_representative_iterations, dir_path, dir_name)

def main():
    """메인 실행 함수."""
    top_level_dir = "."
    all_dirs = [d for d in os.listdir(top_level_dir) if os.path.isdir(os.path.join(top_level_dir, d)) and not d.startswith('.')]
    if not all_dirs:
        print(f"'{top_level_dir}' 안에서 처리할 하위 디렉토리를 찾을 수 없습니다.")
        return
    print(f"Found {len(all_dirs)} director(y/ies) to analyze.")
    for dir_name in sorted(all_dirs):
        process_bandwidth_dir(os.path.join(top_level_dir, dir_name), top_level_dir)

if __name__ == "__main__":
    main()