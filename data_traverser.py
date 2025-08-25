"""遍历目录结构，生成符合条件的时间段数据路径"""
import os
from config import BASE_DIR, COMPONENT_SUFFIXES, SELECTED_STATIONS

def get_all_time_window_paths():
    """
    返回格式：
    [
        {
            "station": "W1111",
            "date": "20250702",
            "time_window": "235959",
            "components": [北南文件路径, 东西文件路径, 垂直文件路径]
        },
        ...
    ]
    """
    all_windows = []

    # 筛选目标台站
    all_stations = [s for s in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, s))]
    target_stations = SELECTED_STATIONS if SELECTED_STATIONS else all_stations

    for station in target_stations:
        station_dir = os.path.join(BASE_DIR, station)
        if not os.path.isdir(station_dir):
            continue

        # 遍历日期目录
        for date in os.listdir(station_dir):
            date_dir = os.path.join(station_dir, date)
            if not os.path.isdir(date_dir):
                continue

            # 遍历时间段目录
            for time_window in os.listdir(date_dir):
                time_window_dir = os.path.join(date_dir, time_window)
                if not os.path.isdir(time_window_dir):
                    continue

                # 匹配三分量文件
                component_files = []
                for suffix in COMPONENT_SUFFIXES:
                    matches = [f for f in os.listdir(time_window_dir)
                               if suffix in f and f.endswith(".SAC")]
                    if len(matches) != 1:
                        print(f"警告: {station}/{date}/{time_window} 缺少{suffix}分量，跳过")
                        break
                    component_files.append(os.path.join(time_window_dir, matches[0]))

                if len(component_files) == 3:  # 确保三分量完整
                    all_windows.append({
                        "station": station,
                        "date": date,
                        "time_window": time_window,
                        "components": component_files
                    })

    print(f"共识别有效时间段: {len(all_windows)}个")
    return all_windows
    