
基于hvsrpy的地震数据处理工具，适用于水平垂直谱比（HVSR）分析，支持批量处理、质量控制与多维度可视化。
<div align="center">
  <img src="https://picsum.photos/800/200?grayscale" alt="HVSR分析示意图" width="70%"/>
  <br>
  <div style="margin: 10px 0; gap: 8px; display: inline-flex;">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" />
    <img src="https://img.shields.io/badge/hvsrpy-2.0.0-green.svg" />
    <img src="https://img.shields.io/badge/License-GPLv3-orange.svg" />
  </div>
</div>


## 作者信息
- **姓名**：彭修文  
- **专业**：中山大学地球物理专业（2022级）  
  


## 版权声明
### 1. 核心依赖版权
本项目基于 **hvsrpy v2.0.0** 开发，其版权归属：  
- 开发者：Joseph P. Vantassel（Virginia Tech）及贡献者  
- 许可证：GNU General Public License v3.0（GPLv3）  
- 官方仓库：[jpvantassel/hvsrpy](https://github.com/jpvantassel/hvsrpy)  

### 2. 本项目版权
© 2024 彭修文（中山大学），基于GPLv3协议开源，保留原作者信息及版权声明即可自由分发与修改。


## 核心功能
- **自动化数据处理**：递归识别台站目录结构，批量处理SAC格式三分量数据
- **双模式处理架构**：
  - 永久性处理：滤波（0.1-23Hz）、去趋势/均值
  - 非永久性处理：窗口分割（可配置时长）、HVSR计算、质量筛选
- **专业级HVSR计算**：采用Konno-Ohmachi平滑算法（带宽系数40），水平分量几何平均
- **多维度质量控制**：基于信噪比（SNR）、波形完整性、频谱特征的自动筛选
- ** publication级可视化**：支持HVSR曲线、三分量波形、多日对比等10+种图表输出


## 处理流程图
<div align="center">
  <img src="images/hvseismic_workflow.png" alt="HVSeismic Tool处理流程图" width="85%" style="border: 1px solid #ddd; padding: 15px; border-radius: 8px;"/>
  <p style="color: #666; font-size: 0.9em; margin-top: 10px;">图1：HVSeismic Tool完整处理流程</p>
</div>


## 处理过程图解
### 1. 窗口分割与质量控制
<div align="center">
  <img src="images/3c_filter_comparison.png" alt="三分量数据质量控制" width="80%" style="border: 1px solid #ddd; padding: 15px; border-radius: 8px;"/>
  <p style="color: #666; font-size: 0.9em; margin-top: 10px;">图2：窗口分割与三分量质量控制图</p>
</div>

### 2. HVSR计算
<div align="center">
  <img src="images/hvsr_curve_example.png" alt="HVSR计算" width="80%" style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-top: 20px;"/>
  <p style="color: #666; font-size: 0.9em; margin-top: 10px;">图3：单时段HVSR计算</p>
</div>

### 3. 台站HVSR曲线统计
<div align="center">
  <img src="images/W27_all_hvsr.png" alt="站HVSR曲线统计" width="80%" style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-top: 20px;"/>
  <p style="color: #666; font-size: 0.9em; margin-top: 10px;">图4：单台站HVSR曲线统计</p>
</div>


## 📂 项目结构hvseismic-tool/
├── main.py                  # 程序入口（协调各模块执行流程）
├── config.py                # 全局参数配置（路径/处理参数/台站列表）
├── data_traverser.py        # 数据路径管理（递归识别台站/日期/时间窗口）
├── permanent_processor.py   # 永久性处理模块（滤波/去趋势）
├── nonpermanent_processor.py # 非永久性处理模块（窗口分割/HVSR计算）
├── result_analyzer.py       # 结果分析与可视化模块（图表生成/统计）
├── logs/                    # 自动生成的处理日志（按日期命名）
└── images/                  # 文档中使用的示意图（流程图/过程图）

## ⚙️ 环境要求
### 1. 基础环境
- Python版本：3.8及以上（推荐3.9版本，兼容性最佳）
- 依赖库安装：
  ```bash
  # 核心依赖（HVSR计算与数据处理）
  pip install hvsrpy==2.0.0 numpy==1.24.3 scipy==1.10.1
  
  # 辅助依赖（可视化与进度显示）
  pip install matplotlib==3.7.1 pandas==1.5.3 tqdm==4.65.0 psutil==5.9.5
  ```

### 2. 输入数据规范
- 格式：SAC格式三分量地震数据（必须包含3个分量）
- 分量标识：BNN（北南）、BNE（东西）、BNZ（垂直）
- 目录结构（严格遵循，否则无法识别）：
  ```
  BASE_DIR/                # 数据根目录（在config.py中配置）
  ├─ 台站ID/（如W27）
  │  ├─ 日期/（如20230101，格式YYYYMMDD）
  │  │  └─ 时间窗口/（如000000，格式HHMMSS）
  │  │     ├─ W27.BNN.20230101.000000.SAC
  │  │     ├─ W27.BNE.20230101.000000.SAC
  │  │     └─ W27.BNZ.20230101.000000.SAC
  ```


## 快速使用指南
### 1. 配置参数
修改`config.py`设置核心路径与处理参数：# 路径配置（示例）
BASE_DIR = "D:/SeismicData"       # 原始数据根目录
PROCESSED_DIR = "D:/ProcessedData" # 永久性处理结果目录
OUTPUT_DIR = "D:/HVSR_Results"     # 最终结果输出目录

# 处理参数（示例）
SELECTED_STATIONS = ["W27", "W13"] # 待处理台站（留空则处理所有）
WINDOW_LENGTH = 30                 # 窗口长度（秒）
FILTER_FREQ = (0.1, 23)            # 滤波频率范围（Hz）


### 2. 运行程序# 进入项目目录
cd hvseismic-tool

# 启动处理
python main.py
### 3. 查看结果
处理完成后，结果按以下结构组织：
1. 永久性处理结果（PROCESSED_DIR）
PROCESSED_DATA/
└── W13/                  # 台站W13
    └── 20250701/         # 2025年7月1日的数据
        ├── 080000_permanent.pkl  # 永久性处理结果（二进制）
        │                         # 内容：经0.1-23Hz带通滤波后的三分量数据
        └── logs/                # 永久性处理日志
            └── 080000.log       # 处理日志：记录采样率（如100Hz）、滤波参数等
2.非永久性处理结果（OUTPUT_DIR）  
HVSR_RESULT/
├── W13/                  # 台站W13
│   └── 20250701/         # 2025年7月1日的数据
│       ├── 080000_hvsr.pkl      # HVSR计算结果（二进制）
│       │                         # 内容：30秒窗口的HVSR振幅、频率、有效窗口掩码
│       ├── 080000_peaks.txt     # 峰值信息文本
│       │                         # 内容示例：
│       │                         # Peak Frequency (Hz): 1.2500
│       │                         # Peak Amplitude: 2.3400
│       │                         # Valid Windows: 15/20
│       │                         # Mean SNR: 3.50 dB
│       └── logs/                # 非永久性处理日志
│           └── 080000.log       # 记录窗口分割（30秒/窗）、质量控制参数等
│
├── plots/                # 可视化结果
│   └── W13/              # 台站W13的图表
│       ├── 20250701/     # 2025年7月1日的图表
│       │   ├── 080000_hvsr.png  # 单窗口HVSR曲线图
│       │   │                    
│       │   └── 080000_3c.png    # 三分量波形图（北南、东西、垂直分量）
│       └── W13_all_hvsr.png     # 台站W13所有时间窗口的HVSR汇总图
│
└── summary/                # 统计数据
    └── W13_summary.csv  # 结构化统计数据
                                              

## 注意事项
1. 首次处理大型数据集时，建议先测试少量数据（通过`SELECTED_STATIONS`指定）
2. 永久性处理结果会保存在`PROCESSED_DIR`，再次运行时可复用，节省时间
3. 所有图表默认以300dpi分辨率导出，满足 publication 要求
4. 日志文件详细记录了每个处理步骤，便于排查数据或参数问题

这是我的第一个项目，后续会继续优化，如果有问题讨论，请联系我！：pengxw8@mail2.sysu.edu.cn / 2663546035@qq.com