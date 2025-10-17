#!/usr/bin/env bash
# 运行 Sheeprl + CarDreamer PPO 实验的启动脚本
# 用法示例： ./run_cardreamer_ppo.sh 1

# 1. 检查参数：需要输入 GPU 卡号
if [ -z "$1" ]; then
    echo "用法: $0 <GPU_ID>"
    echo "例如: $0 1   # 使用 GPU1"
    exit 1
fi
GPU_ID=$1

# 2. 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cardreamer_sheeprl

# 3. 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 4. 避免显存碎片导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 在设置 GPU 之后、进入 sheeprl 之前，插入这行（不会报错）
fuser -k 9000/tcp 2>/dev/null || true


# 5. 准备日志目录
LOG_DIR="$(pwd)/mylog/ppo_debug"
mkdir -p "$LOG_DIR"
DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/ppo_debug_${DATE_STR}.log"

# 6. 进入 sheeprl 目录并启动训练（输出重定向到日志文件）
cd "$(dirname "$0")/sheeprl"
python sheeprl.py \
  exp=ppo_recurrent_cardreamer_stable \
  env.id=FourLane env.wrapper.id=null env.wrapper.task_name=carla_four_lane \
  fabric.accelerator=gpu \
  > "$LOG_FILE" 2>&1

# 7. 提示日志位置
echo "日志已保存到: $LOG_FILE"
