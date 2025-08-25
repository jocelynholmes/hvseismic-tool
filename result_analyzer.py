import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
import logging
import traceback
import psutil
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm 
from config import OUTPUT_DIR, SELECTED_STATIONS, PROCESSED_DIR
import multiprocessing as mp
from multiprocessing.pool import Pool

# 确保matplotlib不使用GUI后端
plt.switch_backend('Agg')

# ------------------------------
# 日志配置
# ------------------------------
def setup_logger():
    logger = logging.getLogger('result_analyzer')
    logger.setLevel(logging.INFO)
    
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'result_analyzer.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


# ------------------------------
# 辅助函数
# ------------------------------
def memory_status():
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024**3),
        'used': mem.used / (1024**3),
        'percent': mem.percent
    }


# ------------------------------
# 单个HVSR曲线绘制（支持跳过功能开关）
# ------------------------------
def plot_hvsr_result(station, date, time_window, output_dir=None, show=False, skip_existing=True):
    """
    绘制单个HVSR曲线
    
    参数:
        skip_existing: 布尔值，是否跳过已存在的图片（True=跳过，False=强制重新绘制）
    """
    # 1. 确定结果文件路径
    result_path = os.path.join(OUTPUT_DIR, station, date, f"{time_window}_hvsr.pkl")
    
    if not os.path.exists(result_path):
        logger.warning(f"结果文件不存在: {result_path}")
        return None, None

    # 2. 确定图片保存路径
    if not output_dir:
        # 若未指定output_dir，默认路径为 OUTPUT_DIR/plots/station/date
        output_dir = os.path.join(OUTPUT_DIR, "plots", station, date)
    # 直接用output_dir拼接文件名（不再重复加station和date）
    save_path = os.path.join(output_dir, f"{time_window}_hvsr.png")
    
    # 3. 图片存在检查（根据skip_existing控制是否执行）
    if skip_existing and os.path.exists(save_path):
        logger.info(f"HVSR图片已存在，跳过绘制: {save_path}")
        # 尝试提取峰值信息
        try:
            with open(result_path, "rb") as f:
                hvsr = pickle.load(f)
            peak_freq, peak_amp = hvsr.mean_curve_peak()
            return peak_freq, peak_amp
        except Exception as e:
            logger.warning(f"获取已存在图片的峰值信息失败: {str(e)}")
            return None, None

    # 4. 绘制图片（若未跳过）
    try:
        with open(result_path, "rb") as f:
            hvsr = pickle.load(f)

        valid_count = hvsr.valid_curve_boolean_mask.sum()
        if valid_count == 0:
            logger.warning(f"有效窗口数为0，跳过绘图: {result_path}")
            return None, None
        
        plt.figure(figsize=(8, 5))

        invalid_lines = []
        valid_lines = []
        for i, amp in enumerate(hvsr.amplitude):
            if not hvsr.valid_curve_boolean_mask[i]:
                line = plt.semilogx(hvsr.frequency, amp, color='gray', alpha=0.3, linewidth=0.5)
                invalid_lines.append(line[0])
            else:
                line = plt.semilogx(hvsr.frequency, amp, color='pink', alpha=0.5, linewidth=0.5)
                valid_lines.append(line[0])

        mean_curve = hvsr.mean_curve()
        std_curve = hvsr.std_curve()
        mean_line = plt.semilogx(hvsr.frequency, mean_curve, 'b-', linewidth=2)
        std_area = plt.fill_between(
            hvsr.frequency,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color='blue', alpha=0.2
        )

        peak_freq, peak_amp = hvsr.mean_curve_peak()
        peak_line = plt.axvline(x=peak_freq, color='r', linestyle='--')

        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('HVSR Amplitude', fontsize=12)
        plt.title(f'HVSR - {station} / {date} / {time_window}', fontsize=14)
        plt.grid(which='both', linestyle='--', alpha=0.5)
        plt.xlim(0.1, 50)
        plt.ylim(0, 6)

        legend_handles = [invalid_lines[0], valid_lines[0], mean_line[0], std_area, peak_line]
        legend_labels = ['Invalid Windows', 'Valid Windows', 'Mean HVSR', '±1 Std Dev', f'Peak: {peak_freq:.2f} Hz']
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=10, loc="upper right")

        stats_text = (f"Peak Frequency: {peak_freq:.4f} Hz\n"
                      f"Peak Amplitude: {peak_amp:.4f}\n"
                      f"Valid Windows: {valid_count}/{hvsr.n_curves}")
        plt.text(0.12, 4.8, stats_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"已保存单窗口HVSR图: {save_path}")

        if not show:
            plt.close()
            del hvsr, invalid_lines, valid_lines, mean_line, std_area, peak_line
            gc.collect()

        return peak_freq, peak_amp

    except Exception as e:
        logger.error(f"绘制单窗口HVSR失败: {result_path} - {str(e)}")
        logger.error(traceback.format_exc())
        return None, None


# ------------------------------
# 批量绘制单个HVSR曲线（支持传递开关参数）
# ------------------------------
def plot_hvsr_wrapper(args):
    # 接收skip_existing参数
    try:
        s, d, tw, output_dir, show, skip_existing = args
        return plot_hvsr_result(s, d, tw, output_dir, show, skip_existing)
    except Exception as e:
        logger.error(f"plot_hvsr_wrapper出错: {str(e)}")
        return None, None


# ------------------------------
# 批量绘制单个HVSR曲线（修复路径重复问题）
# ------------------------------
def batch_plot_hvsr(station=None, date=None, output_dir=None, show=False, batch_size=10, skip_existing=True):
    """批量绘制HVSR曲线，修复路径重复问题"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_dir = output_dir or os.path.join(OUTPUT_DIR, "plots")
    stations = [station] if station else os.listdir(OUTPUT_DIR)
    cpu_count = max(1, psutil.cpu_count() // 2)

    for s_idx in range(0, len(stations), batch_size):
        batch_stations = stations[s_idx:s_idx + batch_size]
        for s in batch_stations:
            station_dir = os.path.join(OUTPUT_DIR, s)
            if not os.path.isdir(station_dir):
                continue

            dates = [date] if date else os.listdir(station_dir)
            for d_idx in range(0, len(dates), batch_size):
                batch_dates = dates[d_idx:d_idx + batch_size]
                for d in batch_dates:
                    date_dir = os.path.join(station_dir, d)
                    if not os.path.isdir(date_dir):
                        continue

                    try:
                        time_windows = [f.split('_')[0] for f in os.listdir(date_dir) if f.endswith('_hvsr.pkl')]
                    except Exception as e:
                        logger.error(f"读取目录 {date_dir} 失败: {str(e)}")
                        continue

                    # 修正：只传递output_dir，不再重复拼接station和date
                    current_output_dir = os.path.join(output_dir, s, d)
                    for tw_idx in range(0, len(time_windows), batch_size):
                        batch_tw = time_windows[tw_idx:tw_idx + batch_size]
                        # 使用try-except包裹进程池操作
                        try:
                            with Pool(cpu_count) as pool:
                                # 只传递current_output_dir，不再重复添加s和d
                                args_list = [(s, d, tw, current_output_dir, show, skip_existing) for tw in batch_tw]
                                # 使用imap代替map，逐个获取结果，减少内存占用
                                for result in pool.imap(plot_hvsr_wrapper, args_list):
                                    pass  # 我们只需要执行，不需要处理结果
                            gc.collect()
                        except BrokenPipeError:
                            logger.warning("检测到BrokenPipeError，正在重新尝试...")
                            # 重试一次
                            with Pool(cpu_count) as pool:
                                args_list = [(s, d, tw, current_output_dir, show, skip_existing) for tw in batch_tw]
                                for result in pool.imap(plot_hvsr_wrapper, args_list):
                                    pass
                            gc.collect()
                        except Exception as e:
                            logger.error(f"批量处理出错: {str(e)}")
                            continue

    logger.info(f"批量绘制完成，结果保存在: {output_dir}")


# ------------------------------
# 导出HVSR统计数据（保持不变）
# ------------------------------
def export_hvsr_stats(station, date, time_window, output_dir=None):
    result_path = os.path.join(OUTPUT_DIR, station, date, f"{time_window}_hvsr.pkl")
    if not os.path.exists(result_path):
        logger.warning(f"结果文件不存在: {result_path}")
        return None

    try:
        with open(result_path, "rb") as f:
            hvsr = pickle.load(f)

        df = pd.DataFrame({
            'Frequency (Hz)': hvsr.frequency,
            'Mean Amplitude': hvsr.mean_curve(),
            'Std Dev Amplitude': hvsr.std_curve()
        })

        output_dir = output_dir or os.path.join(OUTPUT_DIR, "stats")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{station}_{date}_{time_window}_stats.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"已导出统计数据: {output_path}")
        return df

    except Exception as e:
        logger.error(f"导出统计数据失败: {str(e)}")
        return None


def batch_export_stats(station=None, date=None):
    stations = [station] if station else os.listdir(OUTPUT_DIR)
    for s in stations:
        station_dir = os.path.join(OUTPUT_DIR, s)
        if not os.path.isdir(station_dir):
            continue

        dates = [date] if date else os.listdir(station_dir)
        for d in dates:
            date_dir = os.path.join(station_dir, d)
            if not os.path.isdir(date_dir):
                continue

            time_windows = [f.split('_')[0] for f in os.listdir(date_dir) if f.endswith('_hvsr.pkl')]
            for tw in time_windows:
                export_hvsr_stats(s, d, tw)

    logger.info("批量导出统计数据完成")


def summarize_site_results(station):
    """汇总台站结果（添加此函数以解决main.py中的引用问题）"""
    try:
        station_dir = os.path.join(OUTPUT_DIR, station)
        if not os.path.isdir(station_dir):
            logger.warning(f"台站目录不存在: {station_dir}")
            return
        
        # 收集所有日期和时间窗口的峰值信息
        peak_data = []
        dates = [d for d in os.listdir(station_dir) if os.path.isdir(os.path.join(station_dir, d))]
        
        for date in dates:
            date_dir = os.path.join(station_dir, date)
            time_windows = [f.split('_')[0] for f in os.listdir(date_dir) if f.endswith('_hvsr.pkl')]
            
            for tw in time_windows:
                result_path = os.path.join(date_dir, f"{tw}_hvsr.pkl")
                if os.path.exists(result_path):
                    try:
                        with open(result_path, "rb") as f:
                            hvsr = pickle.load(f)
                        peak_freq, peak_amp = hvsr.mean_curve_peak()
                        valid_count = hvsr.valid_curve_boolean_mask.sum()
                        total_count = hvsr.n_curves
                        
                        peak_data.append({
                            'date': date,
                            'time_window': tw,
                            'peak_frequency': peak_freq,
                            'peak_amplitude': peak_amp,
                            'valid_windows': f"{valid_count}/{total_count}"
                        })
                    except Exception as e:
                        logger.error(f"处理 {result_path} 时出错: {str(e)}")
        
        # 保存汇总结果
        if peak_data:
            df = pd.DataFrame(peak_data)
            output_dir = os.path.join(OUTPUT_DIR, "summary")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{station}_summary.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"台站 {station} 汇总结果已保存至: {output_path}")
            
            # 绘制峰值频率随时间变化图
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(df['date'] + ' ' + df['time_window']), df['peak_frequency'], 'ko')
            plt.xlabel('Time')
            plt.ylabel('Peak Frequency (Hz)')
            plt.title(f'Peak Frequency Variation - {station}')
            plt.grid(True)
            plt.xticks()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{station}_peak_freq_trend.png"), dpi=150)
            plt.close()
            
        return peak_data
        
    except Exception as e:
        logger.error(f"汇总台站 {station} 结果失败: {str(e)}")
        return None


# ------------------------------
# 绘制3C分量图（支持跳过功能开关）
# ------------------------------
def plot_3c_components(station, date, time_window, output_dir=None, show=False, skip_existing=True):
    """
    绘制3C分量图（修复有效窗口切片与归一化问题）
    
    参数:
        skip_existing: 布尔值，是否跳过已存在的图片（True=跳过，False=强制重新绘制）
    """
    # 1. 确定结果文件路径
    result_path = os.path.join(OUTPUT_DIR, station, date, f"{time_window}_hvsr.pkl")
    if not os.path.exists(result_path):
        logger.error(f"结果文件不存在: {result_path}")
        return

    # 2. 确定3C图保存路径
    if not output_dir:
        output_dir = os.path.join(OUTPUT_DIR, "plots")
    station_date_dir = os.path.join(output_dir, station, date)
    save_path = os.path.join(station_date_dir, f"{time_window}_3c_components.png")
    
    # 3. 图片存在检查（根据skip_existing控制是否执行）
    if skip_existing and os.path.exists(save_path):
        logger.info(f"3C分量图已存在，跳过绘制: {save_path}")
        return

    # 4. 绘制图片（若未跳过）
    mem_start = memory_status()
    logger.info(f"开始绘制3C图: {station} {date} {time_window}, 内存使用: {mem_start['used']:.2f}GB")

    try:
        with open(result_path, "rb") as f:
            hvsr = pickle.load(f)
        processed_path = os.path.join(PROCESSED_DIR, station, date, f"{time_window}_permanent.pkl")
        if not os.path.exists(processed_path):
            logger.error(f"永久性处理结果文件不存在: {processed_path}")
            return
            
        with open(processed_path, "rb") as f:
            recording = pickle.load(f)

        ns_data, ew_data, vt_data = recording.ns.amplitude, recording.ew.amplitude, recording.vt.amplitude
        fs = 1.0 / recording.ns.dt_in_seconds  # 采样率（Hz）

        # 归一化函数：全局基准（全量数据）
        def normalize(data):
            max_val = np.max(np.abs(data))
            return data / max_val if max_val != 0 else data

        # 提前归一化全量数据（统一基准，避免重复计算）
        normalized_ns = normalize(ns_data)
        normalized_ew = normalize(ew_data)
        normalized_vt = normalize(vt_data)

        window_len = int(30 * fs)  # 30秒对应的采样点数（与非永久性处理窗口长度一致）
        valid_indices = np.where(hvsr.valid_curve_boolean_mask)[0]
        logger.info(f"有效窗口数: {len(valid_indices)}")

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].set_title('North-South Recording', fontsize=12)
        axes[1].set_title('East-West Recording' , fontsize=12)
        axes[2].set_title('Vertical Recording', fontsize=12)

        time_axis = np.arange(len(ns_data)) / fs  # 全局时间轴（秒）

        # 绘制灰色原始数据（复用全局归一化结果）
        axes[0].plot(time_axis, normalized_ns, 'gray', alpha=0.5, label='Original')
        axes[1].plot(time_axis, normalized_ew, 'gray', alpha=0.5, label='Original')
        axes[2].plot(time_axis, normalized_vt, 'gray', alpha=0.5, label='Original')

        # 修复图例：仅首次绘制有效窗口时显示label
        first_accepted = True  
        batch_size = 25  # 批量绘制避免图例重复
        for i in range(0, len(valid_indices), batch_size):
            for idx in valid_indices[i:i+batch_size]:
                start = idx * window_len  # 窗口起始索引
                end = start + window_len   # 窗口结束索引
                win_time = np.arange(start, end) / fs  # 窗口时间轴（秒）
                
                legend_label = 'Accepted' if first_accepted else "_nolegend_"
                # 绘制粉色有效窗口（截取归一化数据的窗口区间）
                axes[0].plot(win_time, normalized_ns[start:end], 'pink', alpha=0.8, linewidth=1.5, label=legend_label)
                axes[1].plot(win_time, normalized_ew[start:end], 'pink', alpha=0.8, linewidth=1.5, label=legend_label)
                axes[2].plot(win_time, normalized_vt[start:end], 'pink', alpha=0.8, linewidth=1.5, label=legend_label)
                
                first_accepted = False  

        # 美化图表
        for ax in axes:
            ax.set_ylabel('Normalized Amplitude', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3, linestyle='--')
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()

        # 确保保存目录存在
        os.makedirs(station_date_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"已保存3C图: {save_path}")
        
        # 内存清理
        if not show:
            plt.close(fig)
            del fig, axes, hvsr, recording
            gc.collect()
        else:
            plt.show()

        mem_end = memory_status()
        logger.info(f"完成绘制3C图: {station} {date} {time_window}, 内存变化: {mem_end['used']-mem_start['used']:.2f}GB")

    except Exception as e:
        logger.error(f"绘制3C图失败: {station} {date} {time_window} - {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        plt.close('all')
        gc.collect()


# ------------------------------
# 绘制单个台站所有HVSR汇总图（保持不变）
# ------------------------------
def plot_station_all_hvsr(station, output_dir=None, show=False, hvsr_data=None):
    station_dir = os.path.join(OUTPUT_DIR, station)
    if not os.path.isdir(station_dir):
        logger.error(f"台站目录不存在: {station_dir}")
        return

    mem_start = memory_status()
    logger.info(f"开始绘制台站 {station} 所有HVSR曲线汇总图, 内存使用: {mem_start['used']:.2f}GB")

    try:
        if hvsr_data is None:
            hvsr_data = []
            dates = [d for d in os.listdir(station_dir) if os.path.isdir(os.path.join(station_dir, d))]
            for date in dates:
                date_dir = os.path.join(station_dir, date)
                time_windows = [f.split('_')[0] for f in os.listdir(date_dir) if f.endswith('_hvsr.pkl')]
                for tw in time_windows:
                    result_path = os.path.join(date_dir, f"{tw}_hvsr.pkl")
                    if os.path.exists(result_path):
                        try:
                            with open(result_path, "rb") as f:
                                hvsr = pickle.load(f)
                                hvsr_data.append((hvsr, date, tw))
                        except Exception as e:
                            logger.error(f"加载HVSR数据失败: {result_path} - {str(e)}")
        
        if not hvsr_data:
            logger.warning(f"未找到台站 {station} 的有效HVSR数据")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        for hvsr, _, _ in hvsr_data:
            if hvsr.valid_curve_boolean_mask is not None:
                valid_indices = np.where(hvsr.valid_curve_boolean_mask)[0]
                for i in valid_indices:
                    ax.semilogx(
                        hvsr.frequency, 
                        hvsr.amplitude[i], 
                        alpha=0.3, linewidth=0.5, color='pink', 
                        label="_nolegend_"
                    )
            else:
                for amp in hvsr.amplitude:
                    ax.semilogx(
                        hvsr.frequency, 
                        amp, 
                        alpha=0.2, linewidth=0.5, color='gray', 
                        label="_nolegend_"
                    )

        all_mean_curves = []
        all_freq_axes = []
        for hvsr, _, _ in hvsr_data:
            if hvsr.valid_curve_boolean_mask is not None and np.any(hvsr.valid_curve_boolean_mask):
                mean_amp = hvsr.mean_curve()
                all_mean_curves.append(mean_amp)
                all_freq_axes.append(hvsr.frequency)
                ax.semilogx(
                    hvsr.frequency, 
                    mean_amp, 
                    'b-', alpha=0.3, linewidth=0.8, 
                    label="_nolegend_"
                )

        legend_handles = []
        legend_labels = []
        if all_mean_curves:
            common_freqs = np.geomspace(0.1, 50, 200)
            interp_means = [
                np.interp(common_freqs, freq_axis, mean_amp, left=np.nan, right=np.nan) 
                for mean_amp, freq_axis in zip(all_mean_curves, all_freq_axes)
            ]
            interp_means = np.array(interp_means)
            valid_mask = ~np.isnan(interp_means).any(axis=0)
            valid_freqs = common_freqs[valid_mask]
            valid_means = interp_means[:, valid_mask]

            if len(valid_freqs) > 0:
                overall_mean = np.mean(valid_means, axis=0)
                overall_std = np.std(valid_means, axis=0)

                mean_line, = ax.semilogx(
                    valid_freqs, overall_mean, 
                    'r-', linewidth=2, 
                    label='Overall Mean'
                )
                legend_handles.append(mean_line)
                legend_labels.append('Overall Mean')

                std_fill = ax.fill_between(
                    valid_freqs, 
                    overall_mean - overall_std, 
                    overall_mean + overall_std, 
                    color='blue', alpha=0.7, 
                    label='±1 Std Dev'
                )
                legend_handles.append(std_fill)
                legend_labels.append('±1 Std Dev')

                peak_idx = np.argmax(overall_mean)
                peak_freq = valid_freqs[peak_idx]
                peak_line = ax.axvline(
                    x=peak_freq, 
                    color='darkred', linestyle='--', linewidth=1.5, 
                    label=f'Avg Peak: {peak_freq:.2f} Hz'
                )
                legend_handles.append(peak_line)
                legend_labels.append(f'Avg Peak: {peak_freq:.2f} Hz')

            single_mean_legend = Line2D(
                [], [], color='blue', alpha=0.3, linewidth=0.8, 
                label='Single Window Mean'
            )
            legend_handles.append(single_mean_legend)
            legend_labels.append('Single Window Mean')

        valid_window_legend = Line2D(
            [], [], color='pink', alpha=0.3, linewidth=0.5, 
            label='Valid Windows'
        )
        legend_handles.append(valid_window_legend)
        legend_labels.append('Valid Windows')

        ax.legend(
            handles=legend_handles, 
            labels=legend_labels, 
            loc='upper right', 
            fontsize=10
        )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('HVSR Amplitude', fontsize=12)
        ax.set_title(f'Station {station} - All HVSR Curves', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.set_xlim(0.1, 50)  
        ax.set_ylim(0.1, 10)  

        if output_dir:
            station_output_dir = os.path.join(output_dir, station)
            os.makedirs(station_output_dir, exist_ok=True)
            save_path = os.path.join(station_output_dir, f"{station}_all_hvsr.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"已保存台站汇总图: {save_path}")
        
        if not show:
            plt.close(fig)
            del fig, ax
            gc.collect()
        else:
            plt.show()

        mem_end = memory_status()
        logger.info(f"完成绘制台站汇总图: {station}, 内存变化: {mem_end['used']-mem_start['used']:.2f}GB")

    except Exception as e:
        logger.error(f"绘制台站汇总图失败: {station} - {str(e)}")
        logger.error(traceback.format_exc())
        plt.close('all')
        gc.collect()