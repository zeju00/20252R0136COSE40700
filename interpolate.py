import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

# --- 설정 ---
# 이 값을 조정하여 대표 곡선의 해상도를 결정합니다.
STANDARD_LENGTH = 500 

# 입력 폴더 (이전 스크립트의 출력)
BASE_LABELED_CSV_DIR = os.path.join("data", "labeled_csvs")

# 최종 결과물(곡선 CSV, 그래프)을 저장할 폴더
OUTPUT_CURVE_DIR = "representative_iteration_curves"

def process_labeled_csv(task_args):
    """
    [병렬 작업용 함수]
    하나의 labeled_power...csv 파일을 읽어, 
    모든 이터레이션을 보간(interpolate)하고 평균을 내어
    하나의 '대표 이터레이션 곡선'을 생성합니다.
    """
    input_csv_path, output_dir = task_args
    pid = os.getpid()
    
    try:
        df = pd.read_csv(input_csv_path)
        if df.empty:
            print(f"  - [PID: {pid}] {os.path.basename(input_csv_path)} is empty. Skipping.")
            return

        # 1. 이터레이션(Fwd->Bwd) 조각들 추출하기
        
        # 'state'가 'forward'로 바뀌는 지점을 이터레이션의 시작으로 감지
        df['state_shifted'] = df['state'].shift(1)
        start_indices = df[(df['state'] == 'forward') & (df['state_shifted'] != 'forward')].index
        
        # 'state'가 'backward'에서 다른 것('idle')으로 바뀌는 지점을 이터레이션의 끝으로 감지
        end_indices = df[(df['state'] == 'backward') & (df['state'].shift(-1) != 'backward')].index

        # 시작/끝 짝 맞추기
        if len(start_indices) > len(end_indices):
            start_indices = start_indices[:len(end_indices)]
        elif len(end_indices) > len(start_indices):
            end_indices = end_indices[len(end_indices) - len(start_indices):]
            
        if len(start_indices) == 0:
            print(f"  - [PID: {pid}] No iterations found in {os.path.basename(input_csv_path)}. Skipping.")
            return

        all_interpolated_curves = []
        all_state_percentages = {'forward': [], 'communication': [], 'backward': []}

        # 2. 보간(Interpolation) 수행
        for start_idx, end_idx in zip(start_indices, end_indices):
            slice_df = df.iloc[start_idx : end_idx + 1]
            
            # (A) 전력 값 보간
            power_values = slice_df['summed_power'].values
            original_x = np.linspace(0, 1, len(power_values))
            target_x = np.linspace(0, 1, STANDARD_LENGTH)
            
            interpolated_power = np.interp(target_x, original_x, power_values)
            all_interpolated_curves.append(interpolated_power)
            
            # (B) 상태(Fwd/Comm/Bwd) 비율 계산 (그래프 시각화용)
            state_counts = slice_df['state'].value_counts()
            total_points = len(slice_df)
            
            all_state_percentages['forward'].append(state_counts.get('forward', 0) / total_points)
            all_state_percentages['communication'].append(state_counts.get('communication', 0) / total_points)
            all_state_percentages['backward'].append(state_counts.get('backward', 0) / total_points)

        if not all_interpolated_curves:
            print(f"  - [PID: {pid}] No valid curves found after processing {os.path.basename(input_csv_path)}. Skipping.")
            return

        # 3. 대표 곡선 계산 (평균)
        representative_curve = np.mean(all_interpolated_curves, axis=0)
        
        # 상태 비율 평균 계산
        avg_fwd_pct = np.mean(all_state_percentages['forward'])
        avg_comm_pct = np.mean(all_state_percentages['communication'])

        # 4. CSV로 저장
        result_df = pd.DataFrame({
            'point_index': range(STANDARD_LENGTH),
            'avg_summed_power': representative_curve
        })
        
        base_name = os.path.basename(input_csv_path).replace('labeled_power_', 'curve_')
        csv_output_path = os.path.join(output_dir, base_name)
        result_df.to_csv(csv_output_path, index=False)

        # 5. 그래프로 저장
        plt.figure(figsize=(12, 6))
        plt.plot(result_df['point_index'], result_df['avg_summed_power'], label=f'Avg. Iteration (n={len(all_interpolated_curves)})')
        
        # 평균 상태 경계 계산
        fwd_end_point = int(avg_fwd_pct * STANDARD_LENGTH)
        comm_end_point = int((avg_fwd_pct + avg_comm_pct) * STANDARD_LENGTH)
        
        # 배경에 상태 표시
        plt.axvspan(0, fwd_end_point, color='sandybrown', alpha=0.3, label='Forward (Avg.)')
        plt.axvspan(fwd_end_point, comm_end_point, color='lightgreen', alpha=0.3, label='Communication (Avg.)')
        plt.axvspan(comm_end_point, STANDARD_LENGTH, color='lightcoral', alpha=0.3, label='Backward (Avg.)')
        
        plt.title(f'Representative Iteration Power Curve\n({base_name.replace(".csv", "")})')
        plt.xlabel(f'Normalized Time (0-{STANDARD_LENGTH} points)')
        plt.ylabel('Average Summed Power (W)')
        plt.legend()
        plt.grid(True)
        
        plot_output_path = os.path.join(output_dir, base_name.replace('.csv', '.png'))
        plt.savefig(plot_output_path)
        plt.close()

        print(f"  -> [PID: {pid}] SUCCESS: Created curve & plot for {os.path.basename(input_csv_path)}")

    except Exception as e:
        print(f"  -> [PID: {pid}] !!! FAILED processing {os.path.basename(input_csv_path)}: {e}")

    return f"Finished {os.path.basename(input_csv_path)}"


def main():
    """
    메인 실행 함수.
    'data/labeled_csvs' 폴더를 스캔하여 모든 CSV에 대해 
    병렬로 '대표 이터레이션' 분석을 수행합니다.
    """
    start_time = time.time()
    
    # 1. 입력 폴더 확인
    if not os.path.isdir(BASE_LABELED_CSV_DIR):
        print(f"Error: Input directory not found: '{BASE_LABELED_CSV_DIR}'")
        print("Please run 'create_labeled_csvs_parallel.py' first.")
        return

    # 2. 출력 폴더 생성
    os.makedirs(OUTPUT_CURVE_DIR, exist_ok=True)
    print(f"All representative curves will be saved to: {OUTPUT_CURVE_DIR}")

    # 3. [병렬화] 모든 'task'를 하나의 리스트로 수집
    tasks_to_process = []
    for root, dirs, files in os.walk(BASE_LABELED_CSV_DIR):
        for file in files:
            if file.startswith("labeled_power_") and file.endswith(".csv"):
                # (입력 CSV 경로, 출력 폴더 경로) 튜플을 작업 목록에 추가
                input_csv_path = os.path.join(root, file)
                
                # 원본 폴더 구조(예: '25M_csvs')를 출력 폴더에도 유지
                relative_path = os.path.relpath(root, BASE_LABELED_CSV_DIR)
                specific_output_dir = os.path.join(OUTPUT_CURVE_DIR, relative_path)
                os.makedirs(specific_output_dir, exist_ok=True)
                
                tasks_to_process.append((input_csv_path, specific_output_dir))

    if not tasks_to_process:
        print(f"No 'labeled_power_...' CSV files found in '{BASE_LABELED_CSV_DIR}'.")
        return

    print(f"\n{'='*50}")
    print(f"Found a total of {len(tasks_to_process)} labeled CSVs to analyze.")
    print(f"Starting parallel interpolation and averaging...")
    print(f"{'='*50}\n")

    # 4. [병렬화] ProcessPoolExecutor를 사용하여 모든 'task'를 병렬로 실행
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_labeled_csv, tasks_to_process))

    end_time = time.time()
    print(f"\n{'='*50}")
    print(f"All analysis complete.")
    print(f"Total {len(results)} files processed in {end_time - start_time:.2f} seconds.")
    print(f"Results are saved in '{OUTPUT_CURVE_DIR}'.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()