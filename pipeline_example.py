#!/usr/bin/env python3
import subprocess
import os
import time
import hashlib
from collections import deque
import itertools
import torch

# 实验参数配置 (根据原始bash脚本)
change_para = {
    "clip-encoder": ["ViT-H-14"],
    "shots": ["1", "2", "4", "8", "16"],
}
unchange_para = {}

experiments_data = list(itertools.product(*change_para.values()))
# SCRIPT_NAME = "eval.py"
SCRIPT_NAME = "sleep.py"
LOG_DIR = "logs"
ALREADY_RAN_DIR = "already_ran"
CONFIG_FILE = "gpu_config.txt"
experiment_id = "xkjdalf"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ALREADY_RAN_DIR, exist_ok=True)
ALREADY_RAN_FILE = os.path.join(ALREADY_RAN_DIR, experiment_id + '.pt')


def parse_config():
    """解析GPU配置文件，返回GPU限制字典"""
    gpu_limits = {}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split(":")
                        if len(parts) == 2:
                            gpu_idx = int(parts[0].strip())
                            max_tasks = int(parts[1].strip())
                            gpu_limits[gpu_idx] = max_tasks
        else:
            print(f"警告: 配置文件 {CONFIG_FILE} 未找到，将使用默认配置")
            # 如果没有配置文件，尝试自动检测GPU数量
            try:
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    gpu_limits[i] = 1  # 默认每个GPU运行1个任务
                print(f"检测到 {num_gpus} 个GPU，使用默认配置")
            except Exception:
                print("无法自动检测GPU。将使用CPU模式运行")
                gpu_limits["cpu"] = 0  # CPU模式下允许多个任务并行
    
    except Exception as e:
        print(f"解析配置文件时出错: {e}")
    
    if not gpu_limits:
        print("无可用GPU配置，将使用CPU模式")
        gpu_limits["cpu"] = 0
    
    return gpu_limits


def generate_experiments():
    """生成所有实验任务（参数组合）"""
    total = len(experiments_data)
    for k, v in change_para.items():
        print(f"{k}: {v}")
    
    # 加载已运行实验集合
    if os.path.exists(ALREADY_RAN_FILE):
        already_ran_set = torch.load(ALREADY_RAN_FILE)
    else:
        already_ran_set = set()
    
    experiments = []
    counter = 0
    
    for exp in experiments_data:
        # 创建唯一的实验ID
        exp_str = '_'.join(str(v) for v in exp)
        exp_id = hashlib.md5(exp_str.encode()).hexdigest()[:8]
        exp_name = f"exp_{exp_str}_{exp_id}"
        
        # 跳过已运行的实验
        if exp_name in already_ran_set:
            continue
            
        # 创建实验配置字典
        exp_dict = {k: v for k, v in zip(change_para.keys(), exp)}
        exp_dict.update({
            'counter': counter,
            'total': total,
            'exp_name': exp_name,
            'exp_name_id': exp_name  # 使用相同ID
        })
        
        experiments.append(exp_dict)
        counter += 1
    
    return experiments, already_ran_set


def run_experiment(exp, gpu_id):
    """运行单个实验任务"""
    # 构建命令
    cmd = ["python", SCRIPT_NAME]
    for k, v in exp.items():
        if k not in ['counter', 'total', 'exp_name', 'exp_name_id']:
            cmd.extend(["--" + k, str(v)])
    for k, v in unchange_para.items():
        cmd.extend(["--" + k, str(v)])
    
    log_path = os.path.join(LOG_DIR, f"{exp['exp_name_id']}.log")
    print(f"[{exp['counter']}/{exp['total']}] 启动任务: {exp['exp_name']}")
    print(f"    GPU: {gpu_id}, 日志: {log_path}")
    
    with open(log_path, "w") as log_file:
        env = os.environ.copy()
        if gpu_id != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        
        process = subprocess.Popen(
            cmd, 
            stdout=log_file, 
            stderr=subprocess.STDOUT, 
            env=env,
            text=True
        )
        return process


def main():
    experiments, already_ran_set = generate_experiments()
    total_tasks = len(experiments)
    
    if total_tasks == 0:
        print("没有需要执行的任务")
        return
    
    # 初始GPU状态
    gpu_usage = {}
    gpu_limits = parse_config()
    for gpu_id, max_tasks in gpu_limits.items():
        gpu_usage[gpu_id] = {
            "max": max_tasks,
            "current": 0,
            "processes": {}
        } 
    print(f"GPU配置: {gpu_limits}")
    
    # 创建任务队列
    task_queue = deque(experiments)
    completed_tasks = 0
    # 存储 (process, gpu_id, exp) 三元组
    active_processes = {}
    
    print(f"\n开始执行 {total_tasks} 个任务...")
    
    # 主调度循环
    try:
        while completed_tasks < total_tasks:
            # 检查并更新GPU配置
            new_limits = parse_config()
            for gpu_id, max_tasks in new_limits.items():
                if gpu_id not in gpu_usage:
                    gpu_usage[gpu_id] = {
                        "max": max_tasks,
                        "current": 0,
                        "processes": {}
                    }
                else:
                    gpu_usage[gpu_id]["max"] = max_tasks
            
            # 检查已完成的任务
            for pid, (process, gpu_id, exp) in list(active_processes.items()):
                if process.poll() is not None:  # 进程已结束
                    # 更新GPU状态
                    if gpu_id in gpu_usage:
                        gpu_usage[gpu_id]["current"] -= 1
                        if pid in gpu_usage[gpu_id]["processes"]:
                            del gpu_usage[gpu_id]["processes"][pid]
                    
                    # 更新任务状态
                    del active_processes[pid]
                    completed_tasks += 1
                    
                    # 添加到已运行集合并保存
                    already_ran_set.add(exp['exp_name_id'])
                    torch.save(already_ran_set, ALREADY_RAN_FILE)
                    
                    print(f"\n[{exp['counter'] + 1}/{exp['total']}] 任务完成: {exp['exp_name']}")
                    print(f"    已完成: {completed_tasks}/{total_tasks} ({(completed_tasks/total_tasks)*100:.1f}%)")
            
            # 尝试启动新任务
            for gpu_id, gpu_data in gpu_usage.items():
                while gpu_data["current"] < gpu_data["max"] and task_queue:
                    exp = task_queue.popleft()
                    process = run_experiment(exp, gpu_id)
                    
                    # 更新GPU状态
                    gpu_data["current"] += 1
                    gpu_data["processes"][process.pid] = exp["exp_name"]
                    
                    # 跟踪活动进程
                    active_processes[process.pid] = (process, gpu_id, exp)
            
            # 显示当前状态
            status_msg = f"状态: 已完成 {completed_tasks}/{total_tasks} | 排队中 {len(task_queue)} | 运行中 {len(active_processes)}"
            gpu_status = []
            for gpu_id, data in gpu_usage.items():
                gpu_status.append(f"GPU{gpu_id}: {data['current']}/{data['max']}")
            
            print(f"\r{status_msg} | {' | '.join(gpu_status)}", end="", flush=True)
            
            # 短暂休眠避免过度消耗CPU
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n检测到中断信号，正在终止所有进程...")
        for pid, (process, _, _) in active_processes.items():
            try:
                process.terminate()
                print(f"已终止进程 PID: {pid}")
            except Exception as e:
                print(f"终止进程 {pid} 时出错: {e}")
        print("所有进程已终止")
        return
    
    print("\n\n所有实验任务已完成！")


if __name__ == "__main__":
    main()
