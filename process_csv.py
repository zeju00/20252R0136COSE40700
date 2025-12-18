import os
import re
import ast
import pandas as pd
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

#
# --- (profile_resource_1031.py에서 가져온 헬퍼 함수 3개) ---
#

def extract_all_timeseries(resource_log_path, dir_name):
    """(원본과 동일) 리소스 로그 파일(resource_*.txt)을 파싱하여 전력 데이터를 추출합니다."""
    try:
        with open(resource_log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"    - [PID: {os.getpid()}] Error reading file {resource_log_path}: {e}")
        return {}
    
    # ... (이하 로직은 원본과 동일) ...
    all_ts_data = {}
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
                            power_data[df_name]['values'].append(value / 1000.0) # mW -> W
                except (ValueError, SyntaxError): continue
        for name, data in power_data.items():
            if data['timestamps']:
                df = pd.DataFrame({'timestamp': data['timestamps'], 'value': data['values']})
                all_ts_data[name] = df
    return all_ts_data

def parse_specific_intervals(content, start_label, end_label):
    """(원본과 동일) (시작, 종료) 타임스탬프 리스트를 추출합니다."""
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

def parse_intervals_flexible(content, start_labels, end_labels):
    """(원본과 동일) 여러 레이블 조합을 시도합니다."""
    for start_label in start_labels:
        for end_label in end_labels:
            intervals = parse_specific_intervals(content, start_label, end_label)
            if intervals: # 하나라도 찾으면 즉시 반환
                print(f"    - [PID: {os.getpid()}] Found intervals using: '{start_label}' -> '{end_label}'")
                return intervals
    return [] # 매칭되는 패턴이 하나도 없음

#
# --- 👇 [핵심 로직: profile_resource_1031.py의 로직을 그대로 구현] ---
#
def process_experiment_task(task_args):
    """
    [병렬 작업용 함수]
    profile_resource_1031.py의 'analyze_experiment' 로직을 기반으로
    (timestamp, summed_power, state) CSV를 생성합니다.
    """
    exp_path, output_csv_dir = task_args
    exp_name = os.path.basename(exp_path)
    pid = os.getpid()
    print(f"- [PID: {pid}] Processing: {exp_name}")

    try:
        # 1. Client/Runner 디렉토리 및 로그 파일 경로 찾기
        client_dir_name = next((d for d in os.listdir(exp_path) if d.startswith("client_") and os.path.isdir(os.path.join(exp_path, d))), None)
        if not client_dir_name:
            print(f"  - [PID: {pid}] Client directory not found. Skipping.")
            return

        client_path = os.path.join(exp_path, client_dir_name)
        runner_dir_name = next((d for d in os.listdir(client_path) if d.startswith("runner_") and os.path.isdir(os.path.join(client_path, d))), None)
        runner_path = os.path.join(client_path, runner_dir_name) if runner_dir_name else None

        client_time_log = os.path.join(client_path, f"time_{client_dir_name}.txt")
        client_resource_log = next((f for f in os.listdir(client_path) if f.startswith("resource_client_")), None)
        
        if not os.path.exists(client_time_log) or not client_resource_log:
            print(f"  - [PID: {pid}] Client time or resource log not found. Skipping.")
            return
            
        with open(client_time_log, 'r', encoding='utf-8') as f:
            client_content = f.read()
            
        runner_content = None
        if runner_path:
            runner_log_path = next((f for f in os.listdir(runner_path) if f.startswith("time_runner_")), None)
            if runner_log_path:
                with open(os.path.join(runner_path, runner_log_path), 'r', encoding='utf-8') as f:
                    runner_content = f.read()

        # 2. 전력 데이터 파싱 (summed_power 계산)
        client_ts_data = extract_all_timeseries(os.path.join(client_path, client_resource_log), exp_name)
        
        df1, df2 = None, None
        if 'cpu_power_w' in client_ts_data and 'gpu_power_w' in client_ts_data:
            df1 = client_ts_data['cpu_power_w']
            df2 = client_ts_data['gpu_power_w']
        elif 'cpu_gpu_power_w' in client_ts_data and 'soc_power_w' in client_ts_data:
            df1 = client_ts_data['cpu_gpu_power_w']
            df2 = client_ts_data['soc_power_w']
        else:
            print(f"  - [PID: {pid}] Could not find required power columns. Skipping summed_power.")
            return

        if df1 is None or df1.empty or df2 is None or df2.empty:
            print(f"  - [PID: {pid}] Power dataframes are empty. Skipping summed_power.")
            return
            
        all_timestamps = sorted(pd.concat([df1['timestamp'], df2['timestamp']]).unique())
        df1_reindexed = df1.set_index('timestamp').reindex(all_timestamps).ffill()
        df2_reindexed = df2.set_index('timestamp').reindex(all_timestamps).ffill()
        summed_values = df1_reindexed['value'] + df2_reindexed['value']
        summed_df = pd.DataFrame({'timestamp': all_timestamps, 'summed_power': summed_values}).dropna()

        # 3. [수정] profile_resource_1031.py의 "모든" 인터벌 파싱 로직
        
        # (!!! 중요: 이 리스트에 사용자의 모든 로그 형식을 추가하세요 !!!)
        FWD_START_LABELS = ["Forward start time", "FWD_START", "Fwd Start"]
        FWD_END_LABELS = ["Forward end time", "FWD_END", "Fwd End"]
        BWD_START_LABELS = ["Backward start time", "BWD_START", "Bwd Start"]
        BWD_END_LABELS = ["Backward end time", "BWD_END", "Bwd End"]
        COMM_START_LABELS = ["Smashed send start time", "COMM_START"]
        COMM_END_LABELS = ["Smashed send end time", "COMM_END"]
        
        # Fwd/Bwd는 Client 로그에만 있음
        forward_intervals = parse_intervals_flexible(client_content, FWD_START_LABELS, FWD_END_LABELS)
        backward_intervals = parse_intervals_flexible(client_content, BWD_START_LABELS, BWD_END_LABELS)
        
        # Comm은 Client/Runner 모두에서 파싱
        client_comm_intervals = parse_intervals_flexible(client_content, COMM_START_LABELS, COMM_END_LABELS)
        communication_intervals = []
        if runner_content:
            runner_comm_intervals = parse_intervals_flexible(runner_content, COMM_START_LABELS, COMM_END_LABELS)
            # profile_resource_1031.py의 min/max 로직
            num_pairs = min(len(client_comm_intervals), len(runner_comm_intervals))
            for i in range(num_pairs):
                c_start, c_end = client_comm_intervals[i]
                r_start, r_end = runner_comm_intervals[i]
                communication_intervals.append((min(c_start, r_start), max(c_end, r_end)))
        else:
            communication_intervals = client_comm_intervals # Runner 없으면 Client 시간만 사용
            
        # 4. 'state' 컬럼 생성 및 "Overlay"
        df = summed_df.copy()
        
        # Epoch 1 시작 시간 찾기
        epoch1_start_match = re.search(r"Epoch 1: Epoch start time\s*=\s*([\d.]+)", client_content)
        if not epoch1_start_match:
            print(f"  - [PID: {pid}] !!! CRITICAL: 'Epoch 1: Epoch start time' not found. Skipping file.")
            return
        
        epoch_start_time = float(epoch1_start_match.group(1))
        
        # Epoch 1 이전 데이터 필터링
        df = df[df['timestamp'] >= epoch_start_time].reset_index(drop=True)
        if df.empty:
            print(f"  - [PID: {pid}] No power data found after Epoch 1 start. Skipping.")
            return
            
        # --- [핵심 수정: Overlay 로직] ---
        # 1. 기본값(배경색)을 'computation'으로 설정
        #    (profile_resource_1031.py의 computation_intervals 로직과 동일)
        df['state'] = 'computation' 
        
        # 2. 'computation' 위에 'communication'을 덮어씀
        for start, end in communication_intervals:
            df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'state'] = 'communication'
            
        # 3. 그 위에 'forward'를 덮어씀
        for start, end in forward_intervals:
            df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'state'] = 'forward'

        # 4. 그 위에 'backward'를 덮어씀
        for start, end in backward_intervals:
            df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'state'] = 'backward'
        # --- [핵심 수정 끝] ---

        # 5. CSV 파일로 저장
        output_filename = os.path.join(output_csv_dir, f"labeled_power_{exp_name}.csv")
        df_final = df[['timestamp', 'summed_power', 'state']]
        
        # (디버깅용)
        f_count = (df_final['state'] == 'forward').sum()
        c_count = (df_final['state'] == 'communication').sum()
        b_count = (df_final['state'] == 'backward').sum()
        comp_count = (df_final['state'] == 'computation').sum() # 'computation' 상태 개수
        
        print(f"  -> [PID: {pid}] SUCCESS: Saved labeled data to {output_filename} (F={f_count}, C={c_count}, B={b_count}, Comp={comp_count})")

    except Exception as e:
        print(f"  -> [PID: {pid}] !!! FAILED processing {exp_name}: {e}")
    
    return f"Finished {exp_name}"

#
# --- (main 함수는 이전과 동일) ---
#
def main():
    """
    메인 실행 함수. (병렬 처리)
    """
    start_time = time.time()
    top_level_dir = "." 
    
    top_level_output_dir = os.path.join(top_level_dir, "data", "labeled_csvs")
    os.makedirs(top_level_output_dir, exist_ok=True)
    print(f"All result folders will be saved inside: {top_level_output_dir}")
    
    exclude_dirs = ['data', 'summed_csvs', 'labeled_csvs', 'lstm_prediction_plots']
    all_dirs = [d for d in os.listdir(top_level_dir) if 
                os.path.isdir(os.path.join(top_level_dir, d)) and 
                not d.startswith('.') and 
                d not in exclude_dirs]
    
    if not all_dirs:
        print(f"'{top_level_dir}'에서 분석할 하위 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"Found {len(all_dirs)} director(y/ies) to analyze.")

    tasks_to_process = []
    for dir_name in sorted(all_dirs):
        dir_path = os.path.join(top_level_dir, dir_name)
        print(f"\n{'='*30} Scanning Directory: {dir_name} {'='*30}")
        
        output_csv_dir = os.path.join(top_level_output_dir, f"{dir_name}_csvs")
        os.makedirs(output_csv_dir, exist_ok=True)
        print(f"Results for this set will be saved to: {output_csv_dir}")
        
        for exp_name in sorted([d for d in os.listdir(dir_path) if d.startswith("learning_sfl-mp_") and os.path.isdir(os.path.join(dir_path, d))]):
            exp_path = os.path.join(dir_path, exp_name)
            tasks_to_process.append((exp_path, output_csv_dir)) 

    print(f"\n{'='*50}")
    print(f"Found a total of {len(tasks_to_process)} experiments to process.")
    print(f"Starting parallel processing using all available CPU cores...")
    print(f"{'='*50}\n")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_experiment_task, tasks_to_process))

    end_time = time.time()
    print(f"\n{'='*50}")
    print(f"All processing complete.")
    print(f"Total {len(results)} tasks finished in {end_time - start_time:.2f} seconds.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()