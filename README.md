# Install requirements
```
pip install -r requirements.txt
```
# run trainer with command
## goemtions
```
python train.py --dataset goemotions --optimizer adamw --model_name multilabel_base_bert --pretrained_bert_name roberta-base --bert_dim 768  --max_seq_len 128 --s 12  --seed 2022 --num_epoch 8 --threshold 0.2 --learning_rate 1e-5
```

## semeval18
```
python train.py --dataset semeval18 --optimizer adamw --model_name multilabel_base_bert --pretrained_bert_name roberta-base  --bert_dim 768 --max_seq_len 128 --s 12  --seed 2022 --num_epoch 8 --threshold 0 --learning_rate 1e-5
```

# run trainer with bat
## In bat file, a random seed that decrements from 2022 is automatically applied
```
SET COUNT=2022
:MyLoop
    IF "%COUNT%" == "2016" GOTO EndLoop
    python train.py --dataset semeval18 --optimizer adamw --model_name multilabel_base_bert --pretrained_bert_name roberta-base  --bert_dim 768 --max_seq_len 128 --s 12  --seed %COUNT% --num_epoch 8 --threshold 0 --learning_rate 1e-5
    SET /A COUNT-=1
    GOTO MyLoop
:EndLoop
```
## So you can simply use bat file with serial random seed
```
.\train_semeval18.bat
.\train_goemotions.bat
.\train_sst5.bat
```

# Result
## It will auto save the log for each epoch,and save the model with minimum loss on validation set and its result on test set.You can find them in root dir after training.
