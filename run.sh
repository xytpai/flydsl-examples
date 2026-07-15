# python -u benchmark_flydsl_vs_inductor_triton.py \
#   --shape 256,4096,4096 \
#   --shape 512,4096,4096 \
#   --shape 1024,4096,4096 \
#   --shape 2048,4096,4096 \
#   --shape 4096,4096,4096 \
#   --dtype fp16 \
#   --backends flydsl-layout,standalone-triton \
#   --warmup 2 \
#   --iters 3 \
#   --check

python -u benchmark_rmsnorm_flydsl_vs_triton.py \
  --shape 16,128 \
  --shape 16,256 \
  --shape 16,512 \
  --shape 16,1024 \
  --shape 16,2048 \
  --dtype bf16 \
  --backends aiter-triton-rmsnorm,flydsl-rmsnorm \
  --warmup 2 \
  --iters 3 \
  --check