# 导出 API 密钥，这一行不需要修改
export E2B_API_KEY=e2b_bd27fe62abed5c9baef572f60d75576a1362c914
export PYTHONPATH=$(pwd):$PYTHONPATH

# 启动 e2b_router.py 保持不变，因为它是单独的脚本
nohup python scripts/e2b_router.py > e2b_router.log 2>&1 &

# 修改 accelerate launch 这一行
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 --main_process_port 29501 \
    --module src.open_r1.grpo --config recipes/config_coupled_code.yaml
