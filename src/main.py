#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
import csv
import KNRM
import CKNRM
import AVGPOOL
import MAXPOOL
import LSTM

import sys
#import json
import logging
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch.autograd import Variable
from DataLoader import DataLoader
from DataLoaderTest import DataLoaderTest

def get_qrels(QRELS_DEV):
    qrels = {}
    with open(QRELS_DEV, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            qid = row[0]
            did = row[2]
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(did)
    return qrels

def data_forward(model, forward_data, output_dir, raw_output):
    result_dict = {}
    writer = open(output_dir,'w')
    fout = open(raw_output, 'w')
    for idx, batch in enumerate(forward_data):
        if idx % 1000 == 0:
            print(idx)
        inputs_q, inputs_d, mask_q, mask_d, docid, qid = batch
        model.eval()
        output = model(inputs_q,inputs_d,mask_q,mask_d)
        output = output.data.tolist()
        # fout.write(str(qid) + '\t' + str(docid) + '\t' + str(output) + '\n')
        tuples = zip(qid, docid, output)
        for item in tuples:
            fout.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')
            if item[0] not in result_dict:
                result_dict[item[0]]=[]
            result_dict[item[0]].append((item[1], item[2])) #{ id: [output] }

    qrels=get_qrels('../data/qrels.dev.tsv')
    no_label = 0
    c_1_j = 0
    c_2_j = 0
    reduce_num = 0
    for qid, value in result_dict.items():
        if qid not in qrels:
            no_label += 1
            continue
        res = sorted(value, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
        count = 0.0
        score = 0.0
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:#if docid in this qrel[qid]'s docid list(which means it is relevant)
                count += 1
                score += count / (i+1) # + pos doc number/total doc num
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:
                c_2_j += 1 / float(i+1)
                break
        if count != 0:
            c_1_j += score / count
        else: # a question without pos doc
            reduce_num += 1

    print(len(result_dict), reduce_num)
    MAP = c_1_j / float(len(result_dict) - no_label)
    MRR = c_2_j / float(len(result_dict) - no_label) #
    #print ""
    #print(" evaluate on " + flag + " MAP: %f" % MAP)
    #print(" evaluate on " + flag + ' MRR: %f' % MRR)
    print(" evaluate on " + " MAP: %f" % MAP)
    print(" evaluate on " + " MRR: %f" % MRR)


    for qid, values in result_dict.items():
        res = sorted(values, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
        for rank,value in enumerate(res):
            writer.write(str(qid)+'\t'+str(value[0])+'\t'+str(rank+1)+'\n')
    # output results:
    print('len of scored dict:',len(result_dict))

def data_evaluate(model, evaluate_data, flag, qrels):
    eval_dict = {}
    c_1_j = 0
    c_2_j = 0
    reduce_num = 0
    for batch in evaluate_data:
        inputs_q, inputs_d, mask_q, mask_d, docid, qid = batch
        model.eval()
        outputs = model(inputs_q, inputs_d, mask_q, mask_d)
        output = outputs.cpu().data.tolist()
        # print(outputs)
        # output = outputs.data.tolist()
        tuples = zip(qid, docid, output)
        for item in tuples:
            if item[0] not in eval_dict: # id not in eval dict
                eval_dict[item[0]] = []
            eval_dict[item[0]].append((item[1], item[2])) # {id: [(docid, output)]}

    no_label = 0
    for qid, value in eval_dict.items():
        if qid not in qrels:
            no_label += 1
            continue
        res = sorted(value, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
        count = 0.0
        score = 0.0
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:#if docid in this qrel[qid]'s docid list(which means it is relevant)
                count += 1
                score += count / (i+1) # + pos doc number/total doc num
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:
                c_2_j += 1 / float(i+1)
                break
        if count != 0:
            c_1_j += score / count
        else: # a question without pos doc
            reduce_num += 1

    print(len(eval_dict), no_label)
    MAP = c_1_j / float(len(eval_dict) - no_label)
    MRR = c_2_j / float(len(eval_dict) - no_label) #
    #print ""
    #print(" evaluate on " + flag + " MAP: %f" % MAP)
    #print(" evaluate on " + flag + ' MRR: %f' % MRR)
    logging.info(f" evaluate on {flag}" + " MAP: %f" % MAP)
    logging.info(f" evaluate on {flag}" + ' MRR: %f' % MRR)
    return MAP, MRR




def train(model, opt, crit, optimizer, train_data, dev_data, test_data):
    ''' Start training '''
    step = 0
    best_map_dev = 0.0
    best_mrr_dev = 0.0
    best_map_test = 0.0
    best_mrr_test = 0.0
    qrels=get_qrels('../data/qrels.dev.tsv')
    for epoch_i in range(opt.epoch):
        total_loss = 0.0
        time_epstart=datetime.now()
        for batch in train_data:
            # prepare data
            inputs_q, inputs_d_pos, inputs_d_neg, mask_q, mask_d_pos, mask_d_neg = batch
            # forward
            optimizer.zero_grad()
            model.train()
            outputs_pos = model(inputs_q, inputs_d_pos, mask_q, mask_d_pos)
            outputs_neg = model(inputs_q, inputs_d_neg, mask_q, mask_d_neg)
            label = torch.ones(outputs_pos.size())#[1,1,1,1...]
            if opt.cuda:
                label = label.cuda()
            batch_loss = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))

            # backward
            batch_loss.backward()

            # update parameters
            optimizer.step()
            step += 1
            total_loss += batch_loss.data[0]
            if opt.is_ensemble:
                if step > 60000:
                    break
            if step % opt.eval_step == 0:
                time_step=datetime.now()-time_epstart
                print(' Epoch %d Training step %d loss %f this epoch time %s' %(epoch_i, step, total_loss,time_step))
                with open(f"{opt.task}.txt", 'a') as logf:
                    logf.write(' Epoch %d Training step %d loss %f this epoch time %s\n' %(epoch_i, step, total_loss,time_step))
                map_dev, mrr_dev = data_evaluate(model, dev_data, "dev", qrels)
                #map_test, mrr_test = data_evaluate(model, test_data, "test")
                # lets just use dev first...so modify like this:
                map_test=map_dev
                mrr_test=mrr_dev

                report_loss = total_loss
                total_loss = 0
                if map_dev >= best_map_dev:
                    best_map_dev = map_dev
                    best_map_test = map_test
                    best_mrr_dev = mrr_dev
                    best_mrr_test = mrr_test
                    print ("best dev-- mrr %f map %f; test-- mrr %f map %f" % (
                    best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                    with open(f"{opt.task}.txt", 'a') as logf:
                            logf.write("best dev-- mrr %f map %f; test-- mrr %f map %f\n" % (
                        best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                else:
                    print("NOT the best dev-- mrr %f map %f; test-- mrr %f map %f" %(mrr_dev,map_dev,mrr_test,map_test))
                    with open(f"{opt.task}.txt", 'a') as logf:
                        logf.write("NOT the best dev-- mrr %f map %f; test-- mrr %f map %f\n" %(mrr_dev,map_dev,mrr_test,map_test))
                if opt.save_model:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'settings': opt,
                        'epoch': epoch_i}
                    if opt.save_mode == 'all':
                        model_name = f'../chkpt/{opt.save_model}' + f'_step_{step}.chkpt'
                        torch.save(checkpoint, model_name)
                    elif opt.save_mode == 'best':
                        model_name = f'../chkpt/{opt.save_model}.chkpt'
                        if map_dev == best_map_dev:
                            best_map_dev = map_dev
                            best_map_test = map_test
                            best_mrr_dev = mrr_dev
                            best_mrr_test = mrr_test
                            with open(f"{opt.task}.txt", 'a') as logf:# record log
                                logf.write(' Epoch %d Training step %d loss %f this epoch time %s' %(epoch_i, step, report_loss,time_step))
                                logf.write("best dev-- mrr %f map %f; test-- mrr %f map %f" %(best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                            torch.save(checkpoint, model_name)
                            print('    - [Info] The checkpoint file has been updated.')
                            with open(f"{opt.task}.txt", 'a') as logf:
                                logf.write('    - [Info] The checkpoint file has been updated.\n')
        time_epend=datetime.now()
        time_ep=time_epend-time_epstart
        print(f'train epoch {str(epoch_i)} using time: {str(time_ep)}')


def kernal_mus(n_kernels):
    """
    get the mu for each gaussian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    l_mu.extend(l_mu[i] - bin_size for i in range(1, n_kernels - 1))
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each gaussian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode',type=str,choices=['train','forward'],default='train')
    parser.add_argument('-train_data')
    parser.add_argument('-val_data')
    parser.add_argument('-test_data')
    parser.add_argument('-embed')
    parser.add_argument('-vocab_size', default=400001, type=int)
    parser.add_argument('-load_model',type=str,default=None)# saved model(chkpt) dir
    parser.add_argument('-task', choices=['KNRM', 'CKNRM', 'MAXPOOL', 'AVGPOOL', 'LSTM'])
    parser.add_argument('-eval_step', type=int, default=1000)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-n_bins', type=int, default=21)
    parser.add_argument('-name', type=int, default=1)
    parser.add_argument('-is_ensemble', type=bool, default=False)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.mu =  kernal_mus(opt.n_bins)
    opt.sigma = kernel_sigmas(opt.n_bins)
    opt.n_layers = 1
    print (opt)
    #with open(opt.task+".txt",'w') as logf:#log file
    #    logf.write(str(opt)+'\n')
    if opt.mode=='train':
        # ========= Preparing DataLoader =========#
        # data_dir='/data/disk1/private/zhangjuexiao/MSMARCOReranking/'
        # train_filename = data_dir+"marco_train_pair_small.pkl"
        #test_filename = data_dir+"marco_eval.pkl"
        # dev_filename = data_dir+"marco_dev.pkl"
        # train_data = pickle.load(open(train_filename, 'rb'))
        #test_data = pickle.load(open(test_filename, 'rb'))

        training_data = DataLoader(
            data=opt.train_data,
            batch_size=opt.batch_size,
            cuda=opt.cuda)

        validation_data = DataLoaderTest(
            data=opt.val_data,
            batch_size=opt.batch_size,
            test=True,
            cuda=opt.cuda)

        test_data=None
        # dev_data = pickle.load(open(dev_filename, 'rb'))

        if opt.task == "KNRM":
            model = KNRM.knrm(opt, opt.embed)
        elif opt.task == "CKNRM":
            model = CKNRM.cknrm(opt, opt.embed)
        elif opt.task == 'AVGPOOL':
            model=AVGPOOL.avgpool(opt, opt.embed)
        elif opt.task == 'MAXPOOL':
            model=MAXPOOL.maxpool(opt, opt.embed)
        elif opt.task == 'LSTM':
            model=LSTM.lstm(opt, opt.embed)
        test_data=None

        crit = nn.MarginRankingLoss(margin=1, size_average=True)

        if opt.cuda:
            model = model.cuda()
            crit = crit.cuda()
        total_time=datetime.now()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
        train(model, opt, crit, optimizer, training_data, validation_data, test_data)
        total_time=datetime.now()-total_time
        print(f'trainning completed, using time: {str(total_time)}')
    elif opt.mode == 'forward':
        print('load pretrained model to forward')
        if opt.load_model is None:
            print('error! specify model!')
            exit()

        chkpt=torch.load(opt.load_model)# load checkpoint
        # opt=chkpt['settings']
        state_dict=chkpt['model']
        state_dict={k:v.cpu() for k,v in state_dict.items()}

        if opt.task == 'KNRM':
            model=KNRM.knrm(opt)
        elif opt.task == 'CKNRM':
            model=CKNRM.cknrm(opt)
        elif opt.task == 'AVGPOOL':
            model=AVGPOOL.avgpool(opt)
        elif opt.task == 'MAXPOOL':
            model=MAXPOOL.maxpool(opt)
        elif opt.task == 'LSTM':
            model=LSTM.lstm(opt)

        model.load_state_dict(state_dict)
        model.cuda()

        test_data = DataLoaderTest(
            data=opt.test_data,
            batch_size=opt.batch_size,
            test=True,
            cuda=opt.cuda)

        data_forward(
            model,
            test_data,
            f'../output/{opt.task}' + '_output_%d.txt' % opt.name,
            f'../output/{opt.task}' + '_raw_output_%d.txt' % opt.name,
        )

if __name__ == "__main__":
    main()
