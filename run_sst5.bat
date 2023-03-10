SET COUNT=2022
:MyLoop
    IF "%COUNT%" == "2016" GOTO EndLoop
    python run.py --dataset sst-5 --optimizer adamw --model_name multilabel_base_bert --pretrained_bert_name roberta-large --bert_dim 1024  --max_seq_len 256 --s 8  --seed %COUNT% --num_epoch 8 --threshold 0.1 --learning_rate 1.5e-6 --batch_size 24
    SET /A COUNT-=1
    GOTO MyLoop
:EndLoop
