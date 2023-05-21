./scripts/clm_run.sh bigscience/bloom-7b1 wikitext wikitext-103-raw-v1 ant-int-flint 4 1 46666 outlier
./scripts/clm_run.sh bigscience/bloom-7b1 c4 realnewslike ant-int-flint 4 1 46666 outlier
./scripts/clm_run.sh facebook/opt-6.7b wikitext wikitext-103-raw-v1 ant-int-flint 4 2 46666 outlier
./scripts/clm_run.sh facebook/opt-6.7b c4 realnewslike ant-int-flint 4 2 46666 outlier
./scripts/clm_run.sh gpt2-xl wikitext wikitext-103-raw-v1 ant-int-flint 4 8 46666 outlier
./scripts/clm_run.sh gpt2-xl c4 realnewslike ant-int-flint 4 8 46666 outlier


CUDA_VISIBLE_DEVICES=0 ./scripts/clm_run.sh bigscience/bloom-7b1 c4 realnewslike ant-int-flint 4 1 46666 outlier &
CUDA_VISIBLE_DEVICES=1 ./scripts/clm_run.sh facebook/opt-6.7b c4 realnewslike ant-int-flint 4 2 46667 outlier &
CUDA_VISIBLE_DEVICES=2 ./scripts/clm_run.sh gpt2-xl c4 realnewslike ant-int-flint 4 8 46668 outlier &