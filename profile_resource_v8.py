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
        plot_filename = os.path.join(output_dir, f"{prefix}_{name}.png")
        try: plt.savefig(plot_filename)
        except Exception as e: print(f"    - Failed to save graph {plot_filename}: {e}")
        plt.close()

def create_aggregate_plots(all_timeseries_collection, output_dir):
    """모든 split_index의 결과를 종합하여 메트릭별 꺾은선 그래프를 생성합니다."""
    metric_labels = {
        'cpu_usage': "Usage (%)", 'cpu_temp': "Temperature (°C)",
        'gpu_temp': "Temperature (°C)", 'gpu_load': "Load (%)",
        'active_cpu_cores': "Active Cores (Count)",
        'cpu_gpu_power_w': "Power (W)", 'soc_power_w': "Power (W)",
        'cpu_power_w': "Power (W)", 'gpu_power_w': "Power (W)",
        'tot_power_w': "Total Power (W)",
        'summed_power_w': "Power (W)"
    }
    print(f"\n{'='*30} Creating Aggregate Plots {'='*30}")
    metrics_data = {}
    for item in all_timeseries_collection:
        for metric_name, df in item['data'].items():
            if metric_name not in metrics_data: metrics_data[metric_name] = []
            metrics_data[metric_name].append({'split_index': item['split_index'], 'df': df, 'start_time': item['start_time']})

    for metric_name, data_list in metrics_data.items():
        plt.figure(figsize=(12, 8))
        for item in sorted(data_list, key=lambda x: x['split_index']):
            si, df, start_time = item['split_index'], item['df'], item['start_time']
            if not df.empty:
                time_sec_relative = df['timestamp'] - start_time
                plt.plot(time_sec_relative, df['value'], label=f'Split Index {si}', linewidth=2)

        title_name = "Summed Power" if metric_name.endswith('summed_power_w') else "Total Power" if metric_name.endswith('tot_power_w') else metric_name.replace("_w", "").replace("_", " ").title()
        plt.title(f'Aggregate {title_name} Across All Split Indexes')
        plt.xlabel("Time (seconds)")
        plt.ylabel(metric_labels.get(metric_name.split('_', 1)[-1], 'Value'))
        plt.grid(True)
        plt.legend()
        plt.xlim(left=0)
        plt.tight_layout()
        agg_plot_filename = os.path.join(output_dir, f"aggregate_{metric_name}.png")
        try:
            plt.savefig(agg_plot_filename)
            print(f"- Aggregate graph saved: {agg_plot_filename}")
        except Exception as e: print(f"- Failed to save aggregate graph {agg_plot_filename}: {e}")
        plt.close()

def calculate_iteration_power_stats(power_df, intervals):
    """Iteration 구간별 전력의 최대/최소값을 계산합니다."""
    if power_df is None or power_df.empty or not intervals:
        return []
    
    stats = []
    for start_time, end_time in intervals:
        iteration_df = power_df[(power_df['timestamp'] >= start_time) & (power_df['timestamp'] <= end_time)]
        if not iteration_df.empty:
            stats.append({
                'start_time': start_time,
                'end_time': end_time,
                'max_power': iteration_df['value'].max(),
                'min_power': iteration_df['value'].min()
            })
    return stats

def plot_graphs_per_epoch(exp_path, exp_name, epoch_intervals, all_data, plot_intervals_dict, iteration_stats_dict):
    """Epoch별로 리소스 사용량 그래프를 생성하고 지정된 디렉토리에 저장합니다."""
    per_epoch_base_dir = os.path.join(exp_path, 'per_epoch')
    print(f"\n  - Creating per-epoch graphs in '{per_epoch_base_dir}'...")

    for i, (epoch_start, epoch_end) in enumerate(epoch_intervals):
        epoch_num = i + 1
        epoch_dir = os.path.join(per_epoch_base_dir, f'epoch_{epoch_num}')
        os.makedirs(epoch_dir, exist_ok=True)

        for prefix, ts_data_collection in all_data.items():
            if not ts_data_collection: continue
            
            avg_max_power, avg_min_power = None, None
            iteration_stats = iteration_stats_dict.get(prefix)
            if iteration_stats:
                epoch_iteration_stats = [s for s in iteration_stats if s['start_time'] >= epoch_start and s['end_time'] <= epoch_end]
                if epoch_iteration_stats:
                    avg_max_power = np.mean([s['max_power'] for s in epoch_iteration_stats])
                    avg_min_power = np.mean([s['min_power'] for s in epoch_iteration_stats])

            epoch_specific_data = {}
            for name, df in ts_data_collection.items():
                epoch_df = df[(df['timestamp'] >= epoch_start) & (df['timestamp'] <= epoch_end)]
                if not epoch_df.empty:
                    epoch_specific_data[name] = epoch_df
            
            if epoch_specific_data:
                epoch_duration = epoch_end - epoch_start
                plot_individual_graphs(
                    epoch_specific_data,
                    output_dir=epoch_dir,
                    exp_name=f"{exp_name}_Epoch_{epoch_num}",
                    prefix=prefix,
                    intervals=plot_intervals_dict,
                    global_start_time=epoch_start,
                    epoch_duration=epoch_duration,
                    avg_max_power=avg_max_power,
                    avg_min_power=avg_min_power
                )

def generate_epoch_reports(exp_path, epoch_intervals, all_data, full_computation_intervals, full_communication_intervals, iteration_stats_dict):
    """Epoch별 통계를 계산하고 CSV 파일로 저장합니다."""
    per_epoch_base_dir = os.path.join(exp_path, 'per_epoch')
    print(f"\n  - Generating per-epoch statistics in '{per_epoch_base_dir}'...")

    for i, (epoch_start, epoch_end) in enumerate(epoch_intervals):
        epoch_num = i + 1
        epoch_dir = os.path.join(per_epoch_base_dir, f'epoch_{epoch_num}')
        os.makedirs(epoch_dir, exist_ok=True)

        epoch_comm_intervals = [(s, e) for s, e in full_communication_intervals if s >= epoch_start and e <= epoch_end]
        epoch_comp_intervals = [(s, e) for s, e in full_computation_intervals if s >= epoch_start and e <= epoch_end]

        epoch_duration = epoch_end - epoch_start
        epoch_comm_duration = sum(e - s for s, e in epoch_comm_intervals)
        epoch_comp_duration = epoch_duration - epoch_comm_duration

        stats = {
            'epoch_total_duration': epoch_duration,
            'epoch_communication_duration': epoch_comm_duration,
            'epoch_computation_duration': epoch_comp_duration,
        }

        for prefix, ts_data in all_data.items():
            if not ts_data: continue

            iteration_stats = iteration_stats_dict.get(prefix)
            if iteration_stats:
                epoch_iteration_stats = [s for s in iteration_stats if s['start_time'] >= epoch_start and s['end_time'] <= epoch_end]
                if epoch_iteration_stats:
                    stats[f'{prefix}_avg_max_iteration_summed_power_w'] = np.mean([s['max_power'] for s in epoch_iteration_stats])
                    stats[f'{prefix}_avg_min_iteration_summed_power_w'] = np.mean([s['min_power'] for s in epoch_iteration_stats])
            
            gpu_load_df = ts_data.get('gpu_load')
            gpu_power_df = ts_data.get('gpu_power_w')
            cpu_gpu_power_df = ts_data.get('cpu_gpu_power_w')
            summed_power_df = ts_data.get('summed_power_w')
            
            stats[f'{prefix}_comm_avg_gpu_load'] = calculate_average_power_in_intervals(gpu_load_df, epoch_comm_intervals)
            stats[f'{prefix}_comm_avg_gpu_power_w'] = calculate_average_power_in_intervals(gpu_power_df, epoch_comm_intervals)
            stats[f'{prefix}_comm_avg_cpu_gpu_power_w'] = calculate_average_power_in_intervals(cpu_gpu_power_df, epoch_comm_intervals)
            stats[f'{prefix}_comm_avg_summed_power_w'] = calculate_average_power_in_intervals(summed_power_df, epoch_comm_intervals)
            
            stats[f'{prefix}_comp_avg_gpu_load'] = calculate_average_power_in_intervals(gpu_load_df, epoch_comp_intervals)
            stats[f'{prefix}_comp_avg_gpu_power_w'] = calculate_average_power_in_intervals(gpu_power_df, epoch_comp_intervals)
            stats[f'{prefix}_comp_avg_cpu_gpu_power_w'] = calculate_average_power_in_intervals(cpu_gpu_power_df, epoch_comp_intervals)
            stats[f'{prefix}_comp_avg_summed_power_w'] = calculate_average_power_in_intervals(summed_power_df, epoch_comp_intervals)
        
        stats_df = pd.DataFrame([stats])
        csv_path = os.path.join(epoch_dir, f'epoch_{epoch_num}_stats.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"    - Saved epoch {epoch_num} statistics to {csv_path}")

def calculate_power_dynamics(power_df, intervals, integral=False):
    """
    구간별 전력의 시작/끝 값을 찾아 증감량(delta), 기울기(slope), 적분값(integral)을 계산합니다.
    """
    if power_df is None or power_df.empty or not intervals:
        return []

    dynamics = []
    timestamps = power_df['timestamp'].to_numpy()
    values = power_df['value'].to_numpy()

    for start_time, end_time in intervals:
        start_idx = np.searchsorted(timestamps, start_time, side='left')
        end_idx = np.searchsorted(timestamps, end_time, side='left')

        if start_idx >= len(timestamps) or end_idx > len(timestamps) or start_idx == end_idx:
            continue
        
        start_power = values[start_idx]
        actual_end_idx = end_idx - 1 if end_idx > 0 else 0
        if actual_end_idx < start_idx: continue
            
        end_power = values[actual_end_idx]
        
        delta = end_power - start_power
        duration = end_time - start_time
        slope = delta / duration if duration > 0 else 0
        
        # 적분 계산 (에너지)
        interval_mask = (timestamps >= start_time) & (timestamps <= end_time)
        interval_ts = timestamps[interval_mask]
        interval_vals = values[interval_mask]
        integral_val = np.trapz(interval_vals, interval_ts) if len(interval_ts) > 1 else 0.0

        dynamics.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'slope': slope,
            'integral': integral_val
        })
    return dynamics

def generate_epoch_dynamics_report(exp_path, epoch_intervals, dynamics_dict, prefix, dynamic_type='slope'):
    """Epoch별로 전력 기울기 또는 적분 데이터를 CSV로 저장하고 막대 그래프를 생성합니다."""
    per_epoch_base_dir = os.path.join(exp_path, 'per_epoch')
    print(f"\n  - Generating per-epoch {dynamic_type} reports in '{per_epoch_base_dir}'...")

    color_map = {
        'forward': 'sandybrown',
        'backward': 'lightcoral',
        'communication': 'lightgreen'
    }
    unit_map = {
        'slope': 'W/s',
        'integral': 'Joules (W*s)'
    }

    for i, (epoch_start, epoch_end) in enumerate(epoch_intervals):
        epoch_num = i + 1
        epoch_dir = os.path.join(per_epoch_base_dir, f'epoch_{epoch_num}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        all_dfs = []
        for name, dynamics in dynamics_dict.items():
            epoch_dynamics = [d for d in dynamics if d['start_time'] >= epoch_start and d['end_time'] <= epoch_end]
            if epoch_dynamics:
                df = pd.DataFrame(epoch_dynamics)
                df['interval_type'] = name
                all_dfs.append(df)

                plt.figure(figsize=(12, 6))
                avg_val = df[dynamic_type].mean()
                iteration_indices = range(1, len(df) + 1)
                plt.bar(iteration_indices, df[dynamic_type], color=color_map.get(name, 'gray'), label=f'Interval {dynamic_type.capitalize()}')
                plt.axhline(y=avg_val, color='red', linestyle='--', label=f'Average {dynamic_type.capitalize()}: {avg_val:.2f}')
                plt.title(f'Epoch {epoch_num} - {prefix.title()} Summed Power {dynamic_type.capitalize()} during {name.title()} Intervals')
                plt.xlabel("Iteration Index")
                plt.ylabel(f"Power {dynamic_type.capitalize()} ({unit_map.get(dynamic_type, '')})")
                
                max_iter = len(df)
                ticks = list(np.arange(5, max_iter + 1, 5))
                if 1 not in ticks and max_iter > 0:
                    ticks.insert(0, 1)
                if len(ticks) > 1:
                    plt.xticks(ticks)

                plt.grid(True, axis='y')
                plt.legend()
                plt.tight_layout()
                plot_filename = os.path.join(epoch_dir, f'{prefix}_epoch_{epoch_num}_{name}_{dynamic_type}s.png')
                plt.savefig(plot_filename)
                plt.close()

        if all_dfs:
            dynamics_df = pd.concat(all_dfs).sort_values(by='start_time').reset_index(drop=True)
            csv_path = os.path.join(epoch_dir, f'epoch_{epoch_num}_{prefix}_{dynamic_type}s.csv')
            dynamics_df.to_csv(csv_path, index=False)
            print(f"    - Saved epoch {epoch_num} {dynamic_type} data to {csv_path}")

def plot_representative_iteration_timeline(representative_data, output_dir, dir_name):
    """Split index별 대표 이터레이션의 전력 사용량 시계열 그래프를 생성합니다."""
    print(f"\n{'='*30} Creating Representative Iteration Timeline Plot {'='*30}")
    
    if not representative_data:
        print("- No data available to create representative iteration timeline plot.")
        return

    plt.figure(figsize=(14, 8))
    
    avg_durations = {'forward': [], 'communication': [], 'backward': []}
    
    for item in representative_data:
        if item['df'].empty: continue
        
        plt.plot(item['df']['time_relative'], item['df']['value'], label=f"Split Index {item['split_index']}")
        
        for name, intervals in item['intervals'].items():
            if intervals:
                avg_durations[name].append(intervals[0][1] - intervals[0][0])
    
    avg_fwd_end = np.mean(avg_durations['forward']) if avg_durations['forward'] else 0
    avg_comm_end = avg_fwd_end + (np.mean(avg_durations['communication']) if avg_durations['communication'] else 0)
    avg_bwd_end = avg_comm_end + (np.mean(avg_durations['backward']) if avg_durations['backward'] else 0)

    plt.axvspan(0, avg_fwd_end, color='sandybrown', alpha=0.2, label='Avg. Forward')
    plt.axvspan(avg_fwd_end, avg_comm_end, color='lightgreen', alpha=0.2, label='Avg. Communication')
    plt.axvspan(avg_comm_end, avg_bwd_end, color='lightcoral', alpha=0.2, label='Avg. Backward')

    plt.title(f'Representative Iteration Power Timeline by Split Index for {dir_name}')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Summed Power (W)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f"representative_iteration_timeline_{dir_name}.png")
    try:
        plt.savefig(plot_filename)
        print(f"- Representative iteration timeline graph saved: {plot_filename}")
    except Exception as e:
        print(f"- Failed to save representative iteration timeline graph: {e}")
    plt.close()
    
def plot_single_representative_iteration(rep_data, output_dir, exp_name):
    """단일 Split Index의 대표 이터레이션 타임라인을 그립니다."""
    print(f"  - Generating representative iteration graph for {exp_name}...")
    
    plt.figure(figsize=(12, 6))
    
    df = rep_data['df']
    intervals = rep_data['intervals']
    
    plt.plot(df['time_relative'], df['value'], marker='o', linestyle='-', markersize=3)
    
    for start, end in intervals.get('forward', []):
        plt.axvspan(start, end, color='sandybrown', alpha=0.5, label='Forward')
    for start, end in intervals.get('communication', []):
        plt.axvspan(start, end, color='lightgreen', alpha=0.5, label='Communication')
    for start, end in intervals.get('backward', []):
        plt.axvspan(start, end, color='lightcoral', alpha=0.5, label='Backward')
        
    plt.title(f'Representative Iteration Power Timeline for {exp_name}')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Summed Power (W)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"{exp_name}_representative_iteration.png")
    try:
        plt.savefig(plot_filename)
        print(f"    - Saved representative iteration graph to {plot_filename}")
    except Exception as e:
        print(f"    - Failed to save representative iteration graph: {e}")
    plt.close()

def analyze_experiment(exp_path, dir_name):
    exp_name = os.path.basename(exp_path)
    split_index_match = re.search(r'_si(\d+)$', exp_name)
    if not split_index_match: return None, None, None, None
    split_index = int(split_index_match.group(1))

    combined_results = {'split_index': split_index}
    client_ts_data, runner_ts_data = {}, {}
    epoch1_start_time = None
    last_epoch_end_time_runner, last_epoch_end_time_client = None, None
    client_comm_intervals, runner_comm_intervals = [], []
    epoch_intervals = []
    forward_intervals, backward_intervals = [], []
    representative_iteration_data = None

    client_dir_name = next((d for d in os.listdir(exp_path) if d.startswith("client_") and os.path.isdir(os.path.join(exp_path, d))), None)
    if not client_dir_name: return None, None, None, None
    client_path = os.path.join(exp_path, client_dir_name)
    client_log_path = os.path.join(client_path, f"time_{client_dir_name}.txt")
    
    if os.path.exists(client_log_path):
        with open(client_log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            combined_results.update({
                'avg_iteration_duration': parse_value(r"Mean Iteration duration\s*=\s*([\d.]+)", content),
                'avg_tokens_per_second': parse_value(r"Tokens per second\s*=\s*([\d.]+)", content),
                'total_training_duration': parse_value(r"Training duration\s*=\s*([\d.]+)", content)
            })
            epoch1_start_match = re.search(r"Epoch 1: Epoch start time\s*=\s*([\d.]+)", content)
            if epoch1_start_match:
                epoch1_start_time = float(epoch1_start_match.group(1))
            
            client_comm_intervals = parse_specific_intervals(content, "Smashed send start time", "Smashed send end time")
            forward_intervals = parse_specific_intervals(content, "Forward start time", "Forward end time")
            backward_intervals = parse_specific_intervals(content, "Backward start time", "Backward end time")
            
            epoch_start_times = [float(t) for t in re.findall(r"Epoch start time\s*=\s*([\d.]+)", content)]
            epoch_end_times = [float(t) for t in re.findall(r"Epoch end time\s*=\s*([\d.]+)", content)]
            if epoch_start_times and len(epoch_start_times) == len(epoch_end_times):
                 epoch_intervals = list(zip(epoch_start_times, epoch_end_times))
            if epoch_end_times:
                last_epoch_end_time_client = max(epoch_end_times)

    runner_dir_name = next((d for d in os.listdir(client_path) if d.startswith("runner_") and os.path.isdir(os.path.join(client_path, d))), None)
    if runner_dir_name:
        runner_path = os.path.join(client_path, runner_dir_name)
        runner_log_path = next((f for f in os.listdir(runner_path) if f.startswith("time_runner_")), None)
        if runner_log_path:
            with open(os.path.join(runner_path, runner_log_path), 'r', encoding='utf-8') as f:
                content = f.read()
                runner_comm_intervals = parse_specific_intervals(content, "Smashed send start time", "Smashed send end time")
                epoch_end_times_runner = [float(t) for t in re.findall(r"Epoch end time\s*=\s*([\d.]+)", content)]
                if epoch_end_times_runner:
                    last_epoch_end_time_runner = max(epoch_end_times_runner)

    communication_intervals = []
    num_pairs = min(len(client_comm_intervals), len(runner_comm_intervals))
    for i in range(num_pairs):
        client_start, client_end = client_comm_intervals[i]
        runner_start, runner_end = runner_comm_intervals[i]
        interval_start = min(client_start, client_end, runner_start, runner_end)
        interval_end = max(client_start, client_end, runner_start, runner_end)
        if interval_end > interval_start:
            communication_intervals.append((interval_start, interval_end))
    if len(client_comm_intervals) != len(runner_comm_intervals):
        print(f"    - Warning: Mismatch in communication events count. Client: {len(client_comm_intervals)}, Runner: {len(runner_comm_intervals)}. Using {num_pairs} pairs.")

    client_resource_file = next((f for f in os.listdir(client_path) if f.startswith("resource_client_")), None)
    if client_resource_file:
        client_ts_data = extract_all_timeseries(os.path.join(client_path, client_resource_file), dir_name)
    if runner_dir_name and 'runner_path' in locals():
        runner_resource_file = next((f for f in os.listdir(runner_path) if f.startswith("resource_runner_")), None)
        if runner_resource_file:
            runner_ts_data = extract_all_timeseries(os.path.join(runner_path, runner_resource_file), dir_name)

    iteration_stats_dict = {}
    
    for ts_data in [client_ts_data, runner_ts_data]:
        if not ts_data: continue
        
        df1, df2 = None, None
        if 'cpu_power_w' in ts_data and 'gpu_power_w' in ts_data:
            df1 = ts_data['cpu_power_w']
            df2 = ts_data['gpu_power_w']
        elif 'cpu_gpu_power_w' in ts_data and 'soc_power_w' in ts_data:
            df1 = ts_data['cpu_gpu_power_w']
            df2 = ts_data['soc_power_w']
            
        if df1 is not None and not df1.empty and df2 is not None and not df2.empty:
            all_timestamps = sorted(pd.concat([df1['timestamp'], df2['timestamp']]).unique())
            df1_reindexed = df1.set_index('timestamp').reindex(all_timestamps).ffill()
            df2_reindexed = df2.set_index('timestamp').reindex(all_timestamps).ffill()
            summed_values = df1_reindexed['value'] + df2_reindexed['value']
            summed_df = pd.DataFrame({'timestamp': all_timestamps, 'value': summed_values}).dropna()
            ts_data['summed_power_w'] = summed_df

    training_end_time = None
    if last_epoch_end_time_client:
        training_end_time = last_epoch_end_time_client
        print(f"    - Using CLIENT's last epoch end time as training end time: {training_end_time}")
    elif last_epoch_end_time_runner:
        training_end_time = last_epoch_end_time_runner
        print(f"    - Warning: Client epoch end time not found. Using RUNNER's last epoch end time: {training_end_time}")
    else:
        all_timestamps_res = [t for ts_data in [client_ts_data, runner_ts_data] if ts_data for df in ts_data.values() if not df.empty for t in df['timestamp'].tolist()]
        if all_timestamps_res:
            training_end_time = max(all_timestamps_res)
            print("    - Warning: Could not find any epoch end times. Falling back to max resource timestamp.")

    if epoch1_start_time and training_end_time and training_end_time > epoch1_start_time:
        new_total_duration = training_end_time - epoch1_start_time
        combined_results['total_training_duration'] = new_total_duration
        print(f"    - Recalculated training duration: {new_total_duration:.2f}s")

    computation_intervals = []
    if epoch1_start_time and training_end_time:
        sorted_comm_intervals = sorted(communication_intervals, key=lambda x: x[0])
        last_event_end = epoch1_start_time
        for comm_start, comm_end in sorted_comm_intervals:
            if comm_start > last_event_end:
                computation_intervals.append((last_event_end, comm_start))
            last_event_end = comm_end
        if training_end_time > last_event_end:
            computation_intervals.append((last_event_end, training_end_time))
    
    if forward_intervals: combined_results['avg_forward_duration'] = np.mean([e - s for s, e in forward_intervals])
    if backward_intervals: combined_results['avg_backward_duration'] = np.mean([e - s for s, e in backward_intervals])
    if communication_intervals: combined_results['avg_comm_duration'] = np.mean([e - s for s, e in communication_intervals])

    for prefix, ts_data in [("client", client_ts_data), ("runner", runner_ts_data)]:
        if not ts_data: continue
        for name, df in ts_data.items():
            if not df.empty:
                combined_results[f'{prefix}_avg_{name}'] = df['value'].mean()
                combined_results[f'{prefix}_max_{name}'] = df['value'].max()
        gpu_load_df = ts_data.get('gpu_load')
        gpu_power_df = ts_data.get('gpu_power_w')
        cpu_gpu_power_df = ts_data.get('cpu_gpu_power_w')
        summed_power_df = ts_data.get('summed_power_w')
        
        combined_results[f'{prefix}_comm_avg_gpu_load'] = calculate_average_power_in_intervals(gpu_load_df, communication_intervals)
        combined_results[f'{prefix}_comm_avg_gpu_power_w'] = calculate_average_power_in_intervals(gpu_power_df, communication_intervals)
        combined_results[f'{prefix}_comm_avg_cpu_gpu_power_w'] = calculate_average_power_in_intervals(cpu_gpu_power_df, communication_intervals)
        combined_results[f'{prefix}_comm_avg_summed_power_w'] = calculate_average_power_in_intervals(summed_power_df, communication_intervals)

        combined_results[f'{prefix}_comp_avg_gpu_load'] = calculate_average_power_in_intervals(gpu_load_df, computation_intervals)
        combined_results[f'{prefix}_comp_avg_gpu_power_w'] = calculate_average_power_in_intervals(gpu_power_df, computation_intervals)
        combined_results[f'{prefix}_comp_avg_cpu_gpu_power_w'] = calculate_average_power_in_intervals(cpu_gpu_power_df, computation_intervals)
        combined_results[f'{prefix}_comp_avg_summed_power_w'] = calculate_average_power_in_intervals(summed_power_df, computation_intervals)
        
        if summed_power_df is not None:
            comm_dynamics = calculate_power_dynamics(summed_power_df, communication_intervals)
            forward_dynamics = calculate_power_dynamics(summed_power_df, forward_intervals)
            backward_dynamics = calculate_power_dynamics(summed_power_df, backward_intervals)

            if comm_dynamics:
                combined_results[f'{prefix}_avg_comm_summed_power_slope'] = np.mean([d['slope'] for d in comm_dynamics])
                combined_results[f'{prefix}_avg_comm_summed_power_integral'] = np.mean([d['integral'] for d in comm_dynamics])
            if forward_dynamics:
                combined_results[f'{prefix}_avg_forward_summed_power_slope'] = np.mean([d['slope'] for d in forward_dynamics])
                combined_results[f'{prefix}_avg_forward_summed_power_integral'] = np.mean([d['integral'] for d in forward_dynamics])
            if backward_dynamics:
                combined_results[f'{prefix}_avg_backward_summed_power_slope'] = np.mean([d['slope'] for d in backward_dynamics])
                combined_results[f'{prefix}_avg_backward_summed_power_integral'] = np.mean([d['integral'] for d in backward_dynamics])
            
            if epoch_intervals:
                dynamics_dict = {'communication': comm_dynamics, 'forward': forward_dynamics, 'backward': backward_dynamics}
                generate_epoch_dynamics_report(exp_path, epoch_intervals, dynamics_dict, prefix, 'slope')
                generate_epoch_dynamics_report(exp_path, epoch_intervals, dynamics_dict, prefix, 'integral')
        
        if prefix == 'client' and summed_power_df is not None:
            num_iters = min(len(forward_intervals), len(backward_intervals))
            if num_iters > 0:
                iteration_intervals = [(forward_intervals[i][0], backward_intervals[i][1]) for i in range(num_iters)]
                iteration_stats_dict[prefix] = calculate_iteration_power_stats(summed_power_df, iteration_intervals)

                iter_durations = np.array([i[1] - i[0] for i in iteration_intervals])
                avg_total_iter_duration = iter_durations.mean()
                rep_idx = (np.abs(iter_durations - avg_total_iter_duration)).argmin()
                
                rep_iter_start, rep_iter_end = iteration_intervals[rep_idx]
                rep_fwd_interval = [f for f in forward_intervals if rep_iter_start <= f[0] < rep_iter_end][0]
                rep_bwd_interval = [b for b in backward_intervals if rep_iter_start < b[1] <= rep_iter_end][0]
                rep_comm_interval = [c for c in communication_intervals if rep_fwd_interval[1] < c[0] and c[1] < rep_bwd_interval[0]][0]

                rep_power_df = summed_power_df[(summed_power_df['timestamp'] >= rep_iter_start) & (summed_power_df['timestamp'] <= rep_iter_end)].copy()
                rep_power_df['time_relative'] = rep_power_df['timestamp'] - rep_iter_start
                
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
    # --- 👇 [수정된 부분] analysis_cols에 integral 항목 추가 ---
    analysis_cols = [
        'comm_avg_gpu_load', 'comm_avg_gpu_power_w', 'comm_avg_cpu_gpu_power_w', 'comm_avg_summed_power_w',
        'comp_avg_gpu_load', 'comp_avg_gpu_power_w', 'comp_avg_cpu_gpu_power_w', 'comp_avg_summed_power_w',
        'avg_forward_summed_power_slope', 'avg_backward_summed_power_slope', 'avg_comm_summed_power_slope',
        'avg_forward_summed_power_integral', 'avg_backward_summed_power_integral', 'avg_comm_summed_power_integral'
    ]
    # --- 👆 [수정된 부분] ---
    
    client_cols = ['client_energy_efficiency'] + [f'client_{stat}_{res}' for res in resource_cols + power_cols for stat in ['avg', 'max']] + [f'client_{col}' for col in analysis_cols]
    
    df = df.reindex(columns=base_cols + client_cols)
    output_csv = os.path.join(top_level_dir, f'experiment_results_{dir_name}.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n-> Analysis for {dir_name} saved to: {output_csv}\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(df)
    
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