for i in `seq $2 $3`;
    do CUDA_VISIBLE_DEVICES=$1 python main.py -train_data ../data/train.txt -val_data ../data/dev_part.txt -task CKNRM -batch_size 64 -save_model CKNRM_$i -vocab_size 315370 -is_ensemble True;
done
