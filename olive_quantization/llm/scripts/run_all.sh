./scripts/clm_run.sh /data/huggingface/opt-6.7b/ wikitext wikitext-103-raw-v1 ant-int-flint 4 2 46666 outlier
./scripts/clm_run.sh facebook/opt-6.7b c4 realnewslike ant-int-flint 4 2 46666 outlier


CUDA_VISIBLE_DEVICES=0 ./scripts/clm_run.sh facebook/opt-6.7b c4 realnewslike ant-int-flint 4 2 46667 outlier &
CUDA_VISIBLE_DEVICES=0 ./scripts/clm_run.sh gpt2-xl c4 realnewslike ant-int-flint 4 8 46668 outlier &