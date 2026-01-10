export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray start --head \
  --num-gpus 8 \
  --port 8266 \
  --dashboard-port 8267
