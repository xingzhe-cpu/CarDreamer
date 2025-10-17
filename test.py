import os
import time
import torch
from torch.distributions.categorical import Categorical
import logging
from datetime import datetime
import numpy as np
import car_dreamer

def make_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = datetime.now().strftime('log_%Y%m%d_%H%M%S.log')
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # 同时输出到控制台（可选）
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)
    return log_path

def random_policy_from_space(action_space):
    """针对 Box(2,) 连续动作空间，生成一个合法动作。"""
    # 最稳妥：直接用空间自带的 sample
    a = action_space.sample()
    # 确保是 (2,) 的 float32 numpy 数组
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    return a

def constant_policy(action_space, accel=0.3, steer=0.0):
    """一个简单的常数策略，方便快速验证环境联通性。"""
    a = np.array([accel, steer], dtype=np.float32)
    # clip 到动作空间范围内（防止越界）
    if hasattr(action_space, "low") and hasattr(action_space, "high"):
        a = np.clip(a, action_space.low, action_space.high)
    return a

def main():
    log_dir = '/data0/user/yejiawei/code/CarDreamer/mylog'
    log_path = make_logger(log_dir)
    logging.info(f'日志文件: {log_path}')

    # 创建环境
    task, _ = car_dreamer.create_task('carla_four_lane')
    obs = task.reset()

    # 打印观测/动作空间
    logging.info(f"观测空间: {task.observation_space}")
    logging.info(f"动作空间: {task.action_space}")


    use_constant_policy = True  # 先强制恒定动作，确认车能动
    forward_accel = 2.0       # 油门强一点，确保能起步
    forward_steps = 200         # 前进200步再切回随机（可选）

    step_i = 0
    while True:
        # 建议先读取动作边界，选择正确的幅度与方向
        low = getattr(task.action_space, "low", None)
        high = getattr(task.action_space, "high", None)

        if use_constant_policy and step_i < forward_steps:
            # 根据动作边界选择合适的常数动作
            if low is not None and high is not None and len(low) == 2:
                # 如果下界第一维是 -1 或更小，则用 +0.6 表示“加速”，负值会刹车
                accel = forward_accel if low[0] < 0 else min(max(forward_accel, low[0]), high[0])
                steer = 0.0
                action = np.array([accel, steer], dtype=np.float32)
                action = np.clip(action, low, high)
            else:
                action = np.array([forward_accel, 0.0], dtype=np.float32)
        else:
            action = random_policy_from_space(task.action_space)

        obs, reward, done, info = task.step(action)

        # 尝试读取速度（不同封装字段可能不同，常见: 'speed', 'velocity', 'linear_speed'）
        speed = None
        if isinstance(info, dict):
            for k in ['speed', 'linear_speed', 'velocity']:
                if k in info:
                    speed = info[k]
                    break

        logging.info(f"[step {step_i}] action={action}, reward={reward}, done={done}, speed={speed}")

        if done:
            logging.info("Episode terminal，重置环境")
            obs = task.reset()
            step_i = 0
            continue

        step_i += 1
        time.sleep(0.03)

if __name__ == "__main__":
    main()