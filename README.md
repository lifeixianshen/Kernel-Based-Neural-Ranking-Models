# Kernel-Based-Neural-Ranking-Models

This is the repository of the codes of the **Neural Kernel Match IR** methods on the [MSMARCO Passage Reranking Task](http://www.msmarco.org/leaders.aspx).

| Rank (Jan 25th 2019) | MSMARCO Passage Re-Ranking         | Eval MRR@10 | Eval MRR@10 |
| -------------------- | ---------------------------------- | ----------- | ----------- |
| 4th                  | Neural Kernel Match IR (Conv-KNRM) | 27.12       | 29.02       |
| 5th                  | Neural Kernel Match IR (KNRM)      | 19.82       | 21.84       |

### Environment Requirement

- Python3

- PyTorch 0.4.1

### Data Download & Preparation

To download and prepare the training data, see [here](https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models/tree/master/data).

### Model

The main codes and running instructions can be found [here](https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models/tree/master/src).

The codes provide models including `KNRM`, `CKNRM`, `MAXPOOL`, `AVGPOOL`, `LSTM`.

- KNRM: [End-to-End Neural Ad-hoc Ranking with Kernel Pooling](https://arxiv.org/abs/1706.06613). Additionally introduced idf information.
- CKNRM: [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](https://dl.acm.org/citation.cfm?doid=3159652.3159659). Additionally introduced idf information.
- MAXPOOL: Calculate the **max** value on the query embedding vectors and document embedding vectors, then use cos_similarity to measure the similarity.
- AVGPOOL: Calculate the **mean** value on the query embedding vectors and document embedding vectors, then use cos_similarity to measure the similarity.
- LSTM: Encode the query embedding vectors and document embedding vectors using RNN, then use cos_similarity to measure the similarity.

### Reproduce the Leaderboard Result

- Neural Kernel Match IR (Conv-KNRM)

  This is the result on ensembling 8 Conv-KNRM models. The code for ensembling is located [here](https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models/src/#ensembling-model).

  Checkpoints can be found [here](https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models/chkpt).


- Neural Kernel Match IR (KNRM)

  This is the result on the KNRM model with `glove.6b.300d` pretrained embedding. You can use the `-embed` option to load the pretrained embedding file:

  ```shell
  # eg.
  CUDA_VISIBLE_DEVICES=0 python main.py -train_data ../data/train.txt -val_data ../data/dev_part.txt -task KNRM -batch_size 64 -save_model CKNRM -vocab_size 315370 -embed ../chkpt/embed.npy
  ```

### Contact

If you have any questions, suggestions or bug reports, please email at qiaoyf96@gmail.com.

