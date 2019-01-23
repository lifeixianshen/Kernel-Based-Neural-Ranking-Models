for i in `seq $2 $3`;
    do CUDA_VISIBLE_DEVICES=$1 python main.py -test_data ../data/dev.txt -task CKNRM -batch_size 512 -load_model CKNRM_$i.chkpt -vocab_size 315370 -mode forward -name $i;
done
