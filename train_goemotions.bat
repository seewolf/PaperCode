SET COUNT=2022
:MyLoop
    IF "%COUNT%" == "2016" GOTO EndLoop
    python train.py --dataset goemotions --optimizer adamw --model_name multilabel_base_bert --pretrained_bert_name roberta-base --bert_dim 768  --max_seq_len 128 --s 12  --seed %COUNT% --num_epoch 8 --threshold 0.2 --learning_rate 1e-5
    SET /A COUNT-=1
    GOTO MyLoop
:EndLoop