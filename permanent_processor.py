"""执行永久性数据处理（降采样、滤波），结果可复用"""
import os
import pickle
import logging
import hvsrpy
from config import get_permanent_settings, PROCESSED_DIR

def setup_permanent_logger(station, date, time_window):
    """设置永久性处理日志"""
    logger = logging.getLogger(f"permanent_{station}_{date}_{time_window}")
    logger.setLevel(logging.INFO)
    log_dir = os.path.join(PROCESSED_DIR, station, date, "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, f"{time_window}.log"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def process_permanent(window_info):
    """
    执行永久性处理：
    1. 加载原始数据
    2. 统一三分量长度
    3. 降采样（若配置）
    4. 滤波（若配置）
    5. 保存处理后的数据（可复用）
    """
    try:
        station = window_info["station"]
        date = window_info["date"]
        time_window = window_info["time_window"]
        component_paths = window_info["components"]
        logger = setup_permanent_logger(station, date, time_window)

        # 输出路径
        output_dir = os.path.join(PROCESSED_DIR, station, date)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{time_window}_permanent.pkl")

        # 若已处理，直接返回
        """
        if os.path.exists(output_path):
            logger.info(f"已存在永久性处理结果，跳过")
            return output_path
        """
        # 1. 加载原始数据
        recording = hvsrpy.read_single(fnames=component_paths)
        dt_in_seconds = recording.ns.dt_in_seconds
        original_rate = 1 / dt_in_seconds
        logger.info(f"加载成功，原始采样率: {original_rate} Hz")

        # 3. 应用永久性处理设置
        settings = get_permanent_settings()

        # 4. 滤波（不可逆）
        if any(settings.filter_corner_frequencies_in_hz):
            recording.butterworth_filter(fcs_in_hz=settings.filter_corner_frequencies_in_hz)
            logger.info(f"滤波完成: {settings.filter_corner_frequencies_in_hz} Hz")

        # 5. 保存永久性处理结果
        with open(output_path, "wb") as f:
            pickle.dump(recording, f)
        logger.info(f"永久性处理完成，保存至: {output_path}")
        return output_path

    except Exception as e:
        # 尝试记录错误日志
        try:
            logger.error(f"处理失败: {str(e)}")
        except:
            print(f"处理 {station}/{date}/{time_window} 失败: {str(e)}")
        return None