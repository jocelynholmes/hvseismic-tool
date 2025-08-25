import multiprocessing as mp
import time
import gc
import psutil
import traceback
from data_traverser import get_all_time_window_paths
from permanent_processor import process_permanent
from nonpermanent_processor import process_nonpermanent
from result_analyzer import batch_plot_hvsr, summarize_site_results, plot_3c_components, plot_station_all_hvsr
import os
import logging
from config import SELECTED_STATIONS, OUTPUT_DIR
import pickle

# 全局日志配置
def setup_global_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件日志
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'main_new.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_all_stations():
    """获取所有存在结果的台站列表"""
    result_dir = os.path.join(os.getenv('OUTPUT_DIR', "/home/mm/短周期地震仪/SAC_DATA/HVSR_RESULT"), "plots")
    try:
        return [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    except Exception as e:
        logger.error(f"扫描台站目录失败: {str(e)}")
        return []

def memory_status():
    """获取当前内存使用状态"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024**3),  # GB
        'used': mem.used / (1024**3),    # GB
        'percent': mem.percent
    }

def main():
    global logger
    logger = setup_global_logger()
    
    # 开始计时
    start_time = time.time()
    logger.info(f"程序启动，系统内存: {memory_status()['percent']:.1f}%")
    
    # 获取所有时间段
    logger.info("===== 扫描数据目录 =====")
    all_windows = get_all_time_window_paths()
    if not all_windows:
        logger.info("未找到有效数据，程序退出")
        return
    
    # 在绘图前判断是否需要自动扫描台站
    stations_to_process = SELECTED_STATIONS or get_all_stations()
    logger.info(f"待处理台站列表: {stations_to_process}")
    
    # 优化内存配置
    batch_size = 25  # 减小批次大小，降低内存压力
    total_batches = (len(all_windows) + batch_size - 1) // batch_size
    cpu_count = mp.cpu_count()
    pool_size = max(1, cpu_count // 3)  # 减小并行进程数，避免内存过载
    
    logger.info(f"系统CPU核心数: {cpu_count}, 使用进程池大小: {pool_size}")
    logger.info(f"每批次处理时间段数量: {batch_size}, 总批次数: {total_batches}")
    
    # 处理所有批次
    for i in range(0, len(all_windows), batch_size):
        current_batch = i // batch_size + 1
        batch_windows = all_windows[i:i+batch_size]
        
        # 检查内存状态
        mem = memory_status()
        logger.info(f"批次 {current_batch}/{total_batches} 开始，内存使用: {mem['used']:.2f}GB ({mem['percent']:.1f}%)")
        
        # 如果内存使用过高，暂停处理
        if mem['percent'] > 85:
            logger.warning(f"内存使用过高: {mem['percent']:.1f}%，暂停处理30秒...")
            time.sleep(30)
        
        # 第一阶段：执行永久性处理
        logger.info(f"\n===== 开始永久性处理（批次 {current_batch}/{total_batches}） =====")
        try:
            with mp.Pool(pool_size) as pool:
                processed_paths = pool.map(process_permanent, batch_windows)
        except Exception as e:
            logger.error(f"永久性处理批次 {current_batch} 失败: {str(e)}")
            continue
        
        # 统计永久性处理结果
        valid_paths = [p for p in processed_paths if p]
        logger.info(f"永久性处理完成: {len(valid_paths)}/{len(batch_windows)} 成功")
        
        # 第二阶段：执行非永久性处理
        logger.info(f"\n===== 开始非永久性处理（批次 {current_batch}/{total_batches}） =====")
        valid_pairs = [(path, win) for path, win in zip(processed_paths, batch_windows) if path]
        
        try:
            with mp.Pool(pool_size) as pool:
                results = pool.starmap(process_nonpermanent, valid_pairs)
        except Exception as e:
            logger.error(f"非永久性处理批次 {current_batch} 失败: {str(e)}")
            continue
        
        # 统计非永久性处理结果
        valid_results = [r for r in results if r]
        logger.info(f"非永久性处理完成: {len(valid_results)}/{len(valid_pairs)} 成功")
        
        # 第三阶段：生成可视化结果
        logger.info(f"\n===== 开始生成可视化结果（批次 {current_batch}/{total_batches}） =====")
        try:
            batch_plot_hvsr(show=False)
        except Exception as e:
            logger.error(f"批量绘制HVSR失败: {str(e)}")
        
        # 绘制3C类图（分批次）
        output_dir = os.path.join(os.getenv('OUTPUT_DIR', "/home/mm/短周期地震仪/SAC_DATA/HVSR_RESULT"), "plots")
        
        # 优化：分批处理台站，避免一次性加载过多数据
        for station in stations_to_process:
            logger.info(f"开始处理台站: {station}")
            
            # 3C绘图优化：分批加载数据
            for window in batch_windows:
                if window["station"] == station:
                    try:
                        plot_3c_components(station, window["date"], window["time_window"], output_dir=output_dir)
                    except Exception as e:
                        logger.error(f"绘制3C类图失败: {station} {window['date']} {window['time_window']} - {str(e)}")
                        logger.error(traceback.format_exc())  # 记录完整堆栈信息
        
        # 释放不再使用的变量并触发垃圾回收
        del processed_paths, valid_paths, valid_pairs, results, valid_results, batch_windows
        gc.collect()
        
        # 再次检查内存状态
        mem = memory_status()
        logger.info(f"批次 {current_batch}/{total_batches} 完成，内存使用: {mem['used']:.2f}GB ({mem['percent']:.1f}%)")
    
    # 收集所有台站的HVSR数据，用于绘制汇总图
    logger.info("\n===== 收集台站HVSR数据 =====")
    output_dir = os.path.join(os.getenv('OUTPUT_DIR', "/home/mm/短周期地震仪/SAC_DATA/HVSR_RESULT"), "plots")
    
    for station in stations_to_process:
        logger.info(f"收集台站 {station} 的HVSR数据")
        
        # 检查台站目录是否存在
        station_dir = os.path.join(OUTPUT_DIR, station)
        if not os.path.isdir(station_dir):
            logger.warning(f"台站目录不存在: {station_dir}")
            continue
        
        # 获取所有日期目录
        dates = [d for d in os.listdir(station_dir) if os.path.isdir(os.path.join(station_dir, d))]
        
        # 存储该台站的所有HVSR数据
        hvsr_data_all = []
        
        # 分批处理日期目录，避免一次性加载过多数据
        date_batch_size = 10
        for date_batch_start in range(0, len(dates), date_batch_size):
            date_batch = dates[date_batch_start:date_batch_start+date_batch_size]
            
            # 加载当前批次的数据
            for date in date_batch:
                date_dir = os.path.join(station_dir, date)
                time_windows = [f.split('_')[0] for f in os.listdir(date_dir) if f.endswith('_hvsr.pkl')]
                
                for tw in time_windows:
                    result_path = os.path.join(date_dir, f"{tw}_hvsr.pkl")
                    if os.path.exists(result_path):
                        try:
                            with open(result_path, "rb") as f:
                                hvsr = pickle.load(f)
                                hvsr_data_all.append((hvsr, date, tw))
                        except Exception as e:
                            logger.error(f"加载HVSR数据失败: {result_path} - {str(e)}")
            
            # 清理当前批次数据
            gc.collect()
        
        # 绘制该台站的所有HVSR汇总图
        if hvsr_data_all:
            try:
                logger.info(f"绘制台站 {station} 的所有HVSR汇总图")
                plot_station_all_hvsr(
                    station, 
                    hvsr_data=hvsr_data_all, 
                    output_dir=output_dir, 
                    show=False
                )
            except Exception as e:
                logger.error(f"绘制台站所有HVSR图失败: {station} - {str(e)}")
                logger.error(traceback.format_exc())
        
        # 释放台站数据
        del hvsr_data_all
        gc.collect()
    
    # 第四阶段：汇总台站结果
    logger.info("\n===== 汇总台站结果 =====")
    for station in stations_to_process:
        try:
            summarize_site_results(station)
        except Exception as e:
            logger.error(f"汇总台站结果失败: {station} - {str(e)}")
            logger.error(traceback.format_exc())
    
    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"\n所有处理完成！总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()