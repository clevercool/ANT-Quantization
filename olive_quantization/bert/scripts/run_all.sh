# BERT_Base GLUE
./scripts/bert_ptq.sh cola base
./scripts/bert_ptq.sh sst2 base
./scripts/bert_ptq.sh mnli base
./scripts/bert_ptq.sh qqp base
./scripts/bert_ptq.sh mrpc base

# BERT_Base SQUAD
./scripts/qa_bert_ptq.sh squad base
./scripts/qa_bert_ptq.sh squad2 base

# BART_Base GLUE
./scripts/bart_ptq.sh cola base
./scripts/bart_ptq.sh sst2 base
./scripts/bart_ptq.sh mnli base
./scripts/bart_ptq.sh qqp base
./scripts/bart_ptq.sh mrpc base

# BART_Base SQUAD
./scripts/qa_bart_ptq.sh squad base
./scripts/qa_bart_ptq.sh squad2 base

# BERT_Large GLUE
./scripts/bert_ptq.sh cola large
./scripts/bert_ptq.sh sst2 large
./scripts/bert_ptq.sh mnli large
./scripts/bert_ptq.sh qqp large
./scripts/bert_ptq.sh mrpc large