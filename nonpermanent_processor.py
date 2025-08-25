"""执行非永久性处理（窗口分割、质量控制等），结果可重新计算"""
import os
import pickle
import logging
import hvsrpy
import numpy as np
from scipy.signal import welch
from config import get_nonpermanent_settings, get_hvsr_settings, OUTPUT_DIR

def setup_nonpermanent_logger(station, date, time_window):
    """设置非永久性处理日志"""
    logger = logging.getLogger(f"nonpermanent_{station}_{date}_{time_window}")
    logger.setLevel(logging.INFO)
    log_dir = os.path.join(OUTPUT_DIR, station, date, "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, f"{time_window}.log"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def calculate_snr(data, noise, fs=None, freq_range=(0.5, 20.0)):
    """计算特定频率范围内的信噪比"""
    f_data, Pxx_data = welch(data, fs, nperseg=256)
    f_noise, Pxx_noise = welch(noise, fs, nperseg=256)

    # 选择特定频率范围内的数据
    mask = (f_data >= freq_range[0]) & (f_data <= freq_range[1])
    signal_power = np.trapz(Pxx_data[mask], f_data[mask])
    noise_power = np.trapz(Pxx_noise[mask], f_noise[mask])

    return signal_power / noise_power

def process_nonpermanent(processed_path, window_info):
    """
    执行非永久性处理：
    1. 加载永久性处理后的数据
    2. 窗口分割、去均值、去趋势（可逆）
    3. 计算HVSR
    4. 手动质量控制（基于振幅和信噪比）
    5. 保存结果
    """
    station = window_info["station"]
    date = window_info["date"]
    time_window = window_info["time_window"]
    logger = setup_nonpermanent_logger(station, date, time_window)

    # 输出路径
    output_dir = os.path.join(OUTPUT_DIR, station, date)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{time_window}_hvsr.pkl")

    try:
        # 1. 加载永久性处理后的数据
        with open(processed_path, "rb") as f:
            recording = pickle.load(f)
        logger.info(f"加载永久性处理数据成功: {processed_path}")

        # 获取实际采样频率
        fs = 1.0 / recording.ns.dt_in_seconds
        logger.info(f"实际采样频率: {fs} Hz")

        # 2. 非永久性预处理（窗口分割、去均值等）
        preprocess_settings = get_nonpermanent_settings()
        preprocessed = hvsrpy.preprocess(records=recording, settings=preprocess_settings)
        logger.info(f"窗口分割完成，共{len(preprocessed)}个窗口")

        # 3. 计算HVSR
        hvsr_settings = get_hvsr_settings()
        hvsr_result = hvsrpy.process(records=preprocessed, settings=hvsr_settings)
        logger.info(f"HVSR计算完成，初始窗口数: {hvsr_result.n_curves}")

        # 4. 手动质量控制（基于振幅和信噪比的双重筛选）
        # 4.1 初始化有效窗口掩码为全True
        hvsr_result.valid_curve_boolean_mask = np.ones(hvsr_result.n_curves, dtype=bool)
        logger.info("手动初始化valid_curve_boolean_mask为全True")

        # 4.2 基于振幅最大值筛选
        amplitudes = hvsr_result.amplitude  # 所有窗口的振幅数据
        max_amplitude_threshold = 10  # 振幅最大值阈值
        window_max_amp = amplitudes.max(axis=1)  # 每个窗口的振幅最大值
        amp_mask = window_max_amp <= max_amplitude_threshold  # 振幅有效掩码
        logger.info(f"振幅筛选后有效窗口数: {amp_mask.sum()}")

        # 4.3 基于峰值频率筛选（手动计算每个窗口的峰值频率）
        peak_frequencies = []
        for amp in amplitudes:
            peak_idx = np.argmax(amp)  # 振幅最大值对应的索引
            peak_freq = hvsr_result.frequency[peak_idx]  # 映射到频率值
            peak_frequencies.append(peak_freq)
        freq_mask = np.array(peak_frequencies) >= 0.5  # 峰值频率≥0.5Hz
        logger.info(f"频率筛选后有效窗口数: {freq_mask.sum()}")

        # 4.4 基于信噪比（SNR）筛选
        snr_mask = []
        snr_values = []
        for record in preprocessed:
            data_east = record.ew  # 东向分量数据
            data_vertical = record.vt  # 垂直分量数据
            data = data_east.amplitude  # 东向分量数值
            noise = data_vertical.amplitude  # 垂直分量数值
            if fs is None:
                fs = 1 / data.dt_in_seconds  # 采样率 = 1/时间步长
                print(f'fs')
            snr = calculate_snr(data, noise, fs)
            snr_values.append(snr)
            snr_mask.append(snr > 1)  # SNR>1为有效
        snr_mask = np.array(snr_mask)

        # 记录信噪比统计信息
        mean_snr = np.mean(snr_values)
        std_snr = np.std(snr_values)
        logger.info(f"信噪比统计: 平均 = {mean_snr:.2f} dB, 标准差 = {std_snr:.2f} dB")
        logger.info(f"信噪比筛选后有效窗口数: {snr_mask.sum()}")

        # 4.5 叠加所有筛选条件（同时满足才有效）
        final_mask = amp_mask & freq_mask & snr_mask
        hvsr_result.valid_curve_boolean_mask = final_mask
        logger.info(f"最终有效窗口数: {final_mask.sum()}")

        # 5. 保存结果
        with open(output_path, "wb") as f:
            pickle.dump(hvsr_result, f)

        # 保存峰值参数
        valid_windows = hvsr_result.valid_curve_boolean_mask.sum()
        total_windows = hvsr_result.n_curves
        with open(os.path.join(output_dir, f"{time_window}_peaks.txt"), "w") as f:
            f.write(f"Peak Frequency (Hz): {hvsr_result.mean_fn_frequency():.4f}\n")
            f.write(f"Peak Amplitude: {hvsr_result.mean_fn_amplitude():.4f}\n")
            f.write(f"Valid Windows: {valid_windows}/{total_windows}\n")
            f.write(f"Mean SNR: {mean_snr:.2f} dB\n")

        logger.info(f"非永久性处理完成，结果保存至: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return None

# 示例调用（需根据实际情况传入参数）
if __name__ == "__main__":
    window_info = {
        "station": "tst1",
        "date": "20250704",
        "time_window": "025959"
    }
    processed_path = os.path.join(OUTPUT_DIR, window_info["station"], window_info["date"], "permanent_processed.pkl")
    process_nonpermanent(processed_path, window_info)