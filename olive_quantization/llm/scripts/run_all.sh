./scripts/clm_run.sh /home/gaozh/llama-7b wikitext wikitext-103-raw-v1 ant-int-flint 4 2 46666 outlier
./scripts/clm_run.sh facebook/opt-6.7b c4 realnewslike ant-int-flint 4 2 46666 outlier

CUDA_VISIBLE_DEVICES=0 ./scripts/clm_run.sh OPT/opt-6.7b wikitext wikitext-103-raw-v1 ant-int-flint 8 2 46666 outlier &
CUDA_VISIBLE_DEVICES=1 ./scripts/clm_run.sh LLAMA/llama-7b c4 realnewslike ant-int-flint 4 2 46666 outlier &
#CUDA_VISIBLE_DEVICES=2 ./scripts/clm_run.sh gpt2-xl c4 realnewslike ant-int-flint 4 8 46668 outlier &