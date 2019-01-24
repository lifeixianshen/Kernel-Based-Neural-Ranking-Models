# Kernel-Based-Neural-Ranking-Models

This is the repository of the codes of the **Neural Kernel Match IR** methods on the MSMARCO Passage Reranking Task.

### Data Download & Preparation

All the data is located in the `/data` folder. You can run the script to automatically download and preprocess the data:

```shell
cd data
./data_preparation.sh
cd ..
```

Within the above script, it downloads the data from MSMARCO website and preprocesses with the `tokenize_train.py` and `tokenize_dev.py`. The code uses the tokenization method provided by the MSMARCO BM25 Baseline, and converts the tokenized terms into indexes according to `vocab.tsv`. 

The `vocab.tsv` was generated on the MSMARCO train & dev set with words appeared at least 5 times. You can also generate your own vocab file via `gen_vocab.py`.

The `idf.norm.tsv` is the normed idf value calculated on the whole MSMARCO train & dev corpora.

### Train & Forward

The main codes are stored in the `src` folder.

The following commands are used for training and testing:

```shell
# Train
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -train_data $TrainFile -val_data $DevFile -task $Option -batch_size $BATCHSIZE -save_model $SaveModelName -vocab_size $VocabSize

# Forward
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -test_data $TestFile -task $Option -batch_size $BATCHSIZE -load_model $SaveModelName.chkpt -vocab_size 315370 -mode forward

# Train eg.
CUDA_VISIBLE_DEVICES=0 python main.py -train_data ../data/train.txt -val_data ../data/dev_part.txt -task CKNRM -batch_size 64 -save_model CKNRM -vocab_size 315370

# Forward eg.
CUDA_VISIBLE_DEVICES=0 python main.py -test_data ../data/dev.txt -task CKNRM -batch_size 512 -load_model CKNRM.chkpt -vocab_size 315370 -mode forward
```

The `$Option` includes `KNRM`, `CKNRM`, `MEANPOOL`, `AVGPOOL`, `LSTM`.

For ensembled model, you should train the model several times (e.g. the submission on conv-KNRM is ensembled with 8 times training). For simplicity, you can use the following scripts to run the training or testing code.

```shell
# Train
./train_ensemble.sh $GPU_ID $RUN_ID_Start $RUN_ID_End # eg ./train_ensemble 0 0 7

# Forward
./eval_ensemble.sh $GPU_ID $RUN_ID_Start $RUN_ID_End # eg ./eval_ensemble 0 0 7
```

Then the `eval_ensemble.py` can merge the results of these models by calculating the average score.