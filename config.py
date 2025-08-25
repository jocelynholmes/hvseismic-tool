"""全局配置参数（区分永久性和非永久性处理参数）"""
from hvsrpy import HvsrPreProcessingSettings, HvsrTraditionalProcessingSettings
import numpy as np
import os

# 路径配置，使用环境变量或默认值
BASE_DIR = os.getenv('BASE_DIR', "/home/mm/短周期地震仪/SAC_DATA")
PROCESSED_DIR = os.getenv('PROCESSED_DIR', "/home/mm/短周期地震仪/SAC_DATA/PROCESSED_DATA")  # 永久性处理结果
OUTPUT_DIR = os.getenv('OUTPUT_DIR', "/home/mm/短周期地震仪/SAC_DATA/HVSR_RESULT")        # 非永久性处理结果
COMPONENT_SUFFIXES = ["BNN", "BNE", "BNZ"]               # 三分量标识
SELECTED_STATIONS = [ "W13","W12","W10","W11","W15","W14","W16","W17","W1","W27","W30","W5","W32(1)","W6","W7"]                   # 目标台站（留空处理所有）

# 永久性处理参数（降采样、滤波等不可逆操作）
def get_permanent_settings():
    return HvsrPreProcessingSettings(
        filter_corner_frequencies_in_hz=[0.1, 23],  # 带通滤波（Hz）
        # 永久性处理不包含窗口分割和去均值（移至非永久性）
        window_length_in_seconds=None,
        # 去趋势（非永久性）
        detrend="linear",
    )

# 非永久性处理参数（窗口分割、质量控制等可逆操作）
def get_nonpermanent_settings():
    return HvsrPreProcessingSettings(
        window_length_in_seconds=30,  # 窗口长度（秒）
        detrend="constant",                  # 去均值（非永久性）
        # 非永久性处理不包含降采样和滤波（已在永久性处理完成）
    )

# HVSR计算参数
def get_hvsr_settings():
    return HvsrTraditionalProcessingSettings(
        smoothing={
            "operator": "konno_and_ohmachi",  # 文档支持的平滑算子
            "bandwidth": 40,  # 文档默认带宽（固定值，非相对带宽）
            "center_frequencies_in_hz": np.geomspace(0.1, 23, 200),  # 包含在smoothing字典内
            "processing_method": "traditional"
        },
        method_to_combine_horizontals="geometric_mean"  # 正确的参数位置
    )