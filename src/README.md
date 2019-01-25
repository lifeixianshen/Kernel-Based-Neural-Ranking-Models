# Kernel-Based-Neural-Ranking-Models

### Train & Forward

The following commands are used for training and testing:

```shell
# Train
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -train_data $TrainFile -val_data $DevFile -task $TaskOption -batch_size $BATCHSIZE -save_model $SaveModelName -vocab_size $VocabSize

# Forward
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -test_data $TestFile -task $Option -batch_size $BATCHSIZE -load_model $SaveModelName.chkpt -vocab_size 315370 -mode forward

# Train eg.
CUDA_VISIBLE_DEVICES=0 python main.py -train_data ../data/train.txt -val_data ../data/dev_part.txt -task CKNRM -batch_size 64 -save_model CKNRM -vocab_size 315370

# Forward eg.
CUDA_VISIBLE_DEVICES=0 python main.py -test_data ../data/dev.txt -task CKNRM -batch_size 512 -load_model CKNRM.chkpt -vocab_size 315370 -mode forward
```

The `$TaskOption` includes `KNRM`, `CKNRM`, `MEANPOOL`, `AVGPOOL`, `LSTM`. Details are explained [here](https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models#Model).

#### Ensembling Model

For ensembled model, you should train the model several times (e.g. the submission on conv-KNRM is ensembled with 8 times training). For simplicity, you can use the following scripts to run the training or testing code.

```shell
# Train
./train_ensemble.sh $GPU_ID $RUN_ID_Start $RUN_ID_End # eg ./train_ensemble 0 0 7

# Forward
./eval_ensemble.sh $GPU_ID $RUN_ID_Start $RUN_ID_End # eg ./eval_ensemble 0 0 7
```

Then the `eval_ensemble.py` can merge the results of these models by calculating the average score.