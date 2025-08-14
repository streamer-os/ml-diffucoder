export E2B_API_KEY=e2b_bd27fe62abed5c9baef572f60d75576a1362c914

# 把本地 repo src 放到 PYTHONPATH 前面（替成你的绝对路径）
export LOCAL_SRC="/WORK/PUBLIC/huangg_work/ljb/ml-diffucoder/src"
export PYTHONPATH="$LOCAL_SRC:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"

nohup python scripts/e2b_router.py > e2b_router.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 --main_process_port 29501 \
    src/open_r1/grpo.py --config recipes/config_coupled_code.yaml
