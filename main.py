import datetime
import json
import math
import sys
import os
import random
import time
import pprint
import string

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from torch import nn
from preprocessor import Preprocessor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
import tqdm
from ast import literal_eval
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler
from torch.utils.data import random_split
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForTokenClassification
from transformers import BertPreTrainedModel
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, AutoModel
import argparse
from modelling import *
from utils.character_utils import get_embed_matrix_and_vocab


def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def custom_print(*msg):
    
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))

class Instructor():

    def __init__(self, tokenizer_checkpoint, train_data_path, eval_data_path, batch_size):

        self.preprocessor = Preprocessor(tokenizer_checkpoint, train_data_path, eval_data_path, batch_size)

        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        


    def token_to_span_map(self,tokens, char_to_token_index):
        
        token_to_span_map = [[0, 0] for idx in range(len(tokens))]

        for i in range(len(char_to_token_index) - 1):
            if char_to_token_index[i] != char_to_token_index[i + 1]:
                if char_to_token_index[i] >= 0:
                    token_to_span_map[char_to_token_index[i]][1] = i + 1

                if char_to_token_index[i + 1] >= 0:
                    token_to_span_map[char_to_token_index[i+1]][0] = i + 1

        return token_to_span_map

    def get_prediction_index(self,ans_labels, span_map):
        
        acronyms = []
        long_forms = []
        for i in range(min(len(span_map), 512)):
            if ans_labels[i] == 1:
                pointer = i
                pointer += 1
                while(pointer < len(ans_labels) and ans_labels[pointer] == 2):
                    pointer += 1
                
                pointer -= 1
                acronyms.append([span_map[i][0], span_map[min(pointer, len(span_map)-1)][1]])



            elif ans_labels[i] == 3:

                pointer = i
                pointer += 1
                while(pointer < len(ans_labels) and ans_labels[pointer] == 4):
                    pointer += 1
                
                pointer -= 1
                long_forms.append([span_map[i][0], span_map[min(pointer, len(span_map)-1)][1]])

        return acronyms, long_forms

    def score_phrase_level(self,key, predictions, verbos=False):
        gold_shorts = set()
        gold_longs = set()
        pred_shorts = set()
        pred_longs = set()

        def find_phrase(seq, shorts, longs):
            for i, data in enumerate(seq):
                for sh in data['acronyms']:
                    shorts.add(str(i)+'#'+str(sh[0])+'-'+str(sh[1]))
                for lf in data['long-forms']:
                    longs.add(str(i)+'#'+str(lf[0])+'-'+str(lf[1]))

        find_phrase(key, gold_shorts, gold_longs)
        find_phrase(predictions, pred_shorts, pred_longs)

        def find_prec_recall_f1(pred, gold):
            correct = 0
            for phrase in pred:
                if phrase in gold:
                    correct += 1
            # print(correct)
            prec = correct / len(pred) if len(pred) > 0 else 1
            recall = correct / len(gold) if len(gold) > 0 else 1
            f1 = 2 * prec * recall / (prec + recall) if prec+recall > 0 else 0
            return prec, recall, f1

        prec_short, recall_short, f1_short = find_prec_recall_f1(pred_shorts, gold_shorts)
        prec_long, recall_long, f1_long = find_prec_recall_f1(pred_longs, gold_longs)
        precision_micro, recall_micro, f1_micro = find_prec_recall_f1(pred_shorts.union(pred_longs), gold_shorts.union(gold_longs))

        precision_macro = (prec_short + prec_long) / 2
        recall_macro = (recall_short + recall_long) / 2
        f1_macro = 2*precision_macro*recall_macro/(precision_macro+recall_macro) if precision_macro+recall_macro > 0 else 0

        if verbos:
            custom_print('Shorts: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(prec_short, recall_short, f1_short))
            custom_print('Longs: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(prec_long, recall_long, f1_long))
            custom_print('micro scores: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(precision_micro, recall_micro, f1_micro))
            custom_print('macro scores: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(precision_macro, recall_macro, f1_macro))

        return precision_macro, recall_macro, f1_macro

    def evaluate_classifier(self,test_dataloader, model, dataset_, tokenizer, eval_data_path):
        
        model.eval()
        # y_preds, y_test = np.array([]), np.array([])
        all_preds = []
        total = 0
        correct = 0
        pred = []
        label = []

        for step, batch in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        
            with torch.no_grad():
                b_input_ids=batch['input_ids'].long().to('cuda')
                b_attn_mask=batch['attention_mask'].long().to('cuda')
                b_labels = batch['labels'].long().to('cuda')
                
                outputs = model(b_input_ids, b_attn_mask, b_labels)
                # print('done')
                
                predictions = outputs[0]
            
            
                predictions = (predictions.cpu().numpy().tolist())

                all_preds.extend(predictions)
            #print(step)

        val_predictions = []

        custom_print("Starting preparation of output json........")


        
        for i in range(len(dataset_)):

            output_dict = {}
            sample = dataset_[i]
            sample['text'] = sample['text'].replace('—', '-')
            output_dict['text'] = sample['text']
            tokens = self.preprocessor.tokenizer(sample['text'], return_offsets_mapping=True)

            
            ans_labels = all_preds[i]
            
            acronyms, long_forms = self.get_prediction_index(ans_labels, tokens['offset_mapping'])
            output_dict['acronyms'] = acronyms
            output_dict['long-forms'] = long_forms
            output_dict['ID'] = str(i + 1)

            val_predictions.append(output_dict)
        


        with open(os.path.join(trg_folder, 'val_output.json'), 'w') as f:
            json.dump(val_predictions, f, indent = 4)

        with open(eval_data_path) as file:
            gold = dict([(d['ID'], {'acronyms':d['acronyms'],'long-forms':d['long-forms']}) for d in json.load(file)])
        
        with open(os.path.join(trg_folder, 'val_output.json')) as file:
            pred = dict([(d['ID'], {'acronyms':d['acronyms'],'long-forms':d['long-forms']}) for d in json.load(file)])
        
        pred = [pred[k] for k,v in gold.items()]
        gold = [gold[k] for k,v in gold.items()]
        p, r, f1 = self.score_phrase_level(gold, pred, verbos=True)
        
        return p, r, f1

    def get_optimizer_grouped_parameters(self,model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    def get_optimizer_scheduler(self,model, train_dataloader):

        total_steps = len(train_dataloader) * num_train_epochs
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps = 1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

        return optimizer, scheduler

    def save_bert_model(self,model):

        torch.save(model.state_dict(), 'best_model.pt')

    def load_model(self,new_checkpoint):
        model = self.get_model(model_id)
        model.to('cuda')
        model.load_state_dict(torch.load(new_checkpoint))
        model.eval()

        return model

    def get_model(self,model_id):

        print(model_id)

        if model_id == 0:
            return Simple_BERT(self.preprocessor.config, model_checkpoint)

        if model_id == 1:

            char_vocab, embed_matrix = get_embed_matrix_and_vocab(self.preprocessor.eval_bert_dataset, 
                                                                    self.preprocessor.train_bert_dataset,
                                                                    self.preprocessor.tokenizer)
            return Transform_CharacterBERT(self.preprocessor.config, model_checkpoint, char_vocab, 
                                            embed_matrix, self.preprocessor.tokenizer, max_word_len, conv_filter_size)
        # if model_id == 3:
        #     return TwoStepAttention()





    def train(self, model, optimizer, scheduler):

        best_macro_f1_val = -1

        for epoch in range(num_train_epochs):
            custom_print("Epoch: " + str(epoch + 1) + ' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            accumulated_loss = 0
            model.train()
            for step, batch in tqdm.tqdm(enumerate(self.preprocessor.train_dataloader), total=len(self.preprocessor.train_dataloader)):
            
            #   print("Starting step :------------------", step)
                b_input_ids=batch['input_ids'].long().to('cuda')
                b_attn_mask=batch['attention_mask'].long().to('cuda')
                b_labels = batch['labels'].long().to('cuda')

                outputs = model(input_ids = b_input_ids, attn_mask = b_attn_mask, labels = b_labels)

                loss = outputs[1]

                accumulated_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            custom_print("Loss on Train Data ...  ", accumulated_loss)


            # custom_print("Running Eval on Training Data after Epoch ............................., ", str(epoch + 1))
            # trainP, trainR, trainF = self.evaluate_classifier(self.preprocessor.train_dataloader, model, self.preprocessor.train_dataset_raw , self.preprocessor.tokenizer, self.train_data_path )

            custom_print("Running Eval on Validation Data after Epoch ............................., ", str(epoch + 1))
            evalP, evalR, evalF = self.evaluate_classifier(self.preprocessor.eval_dataloader, model, self.preprocessor.eval_dataset_raw , self.preprocessor.tokenizer, self.eval_data_path )
            
            custom_print("Validation Results #########################: P{}   R{}    F{} after Epoch {}".format( evalP, evalR, evalF, str(epoch + 1)))
            
            
            
            
            if evalF > best_macro_f1_val:         

                best_macro_f1_val = evalF   

                evalP, evalR, evalF = self.evaluate_classifier(self.preprocessor.eval_dataloader, model, self.preprocessor.eval_dataset_raw, self.preprocessor.tokenizer, self.eval_data_path)



                self.save_bert_model(model)
                
            custom_print('\n')

            
        custom_print("Done!")

        custom_print("\n\n")
        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--src_folder', type=str, default="data/")
    parser.add_argument('--trg_folder', type=str, default="logs/")
    parser.add_argument('--job_mode', type=str, default="train")
    parser.add_argument('--model_id', type=int, default=0) ##needed
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=6) ##needed
    parser.add_argument('--seed_value', type = int, default = 42)
    parser.add_argument('--tokenizer_checkpoint', type = str, default = '') ##needed
    parser.add_argument('--model_checkpoint', type = str, default='') ##needed
    parser.add_argument('--dataset', type = str, default = 'english/legal') ##needed
    parser.add_argument('--log_file', type = str, default = 'training.log')
    parser.add_argument('--max_word_len', type = int, default = 16) ### when model_id = 1
    parser.add_argument('--cnn_filter_size', type = int, default = 4) ## when model_id = 1

    args = parser.parse_args()

    seed_value = args.seed_value
    num_train_epochs = args.epoch
    src_folder = args.src_folder
    trg_folder = args.trg_folder
    tokenizer_checkpoint = args.tokenizer_checkpoint
    model_checkpoint = args.model_checkpoint
    bs = args.batch_size
    dataset_folder = args.dataset
    log_file = args.log_file

    train_data_path = os.path.join(src_folder, dataset_folder, 'train.json')
    eval_data_path = os.path.join(src_folder, dataset_folder, 'dev.json' )

    ins = Instructor(tokenizer_checkpoint, train_data_path, eval_data_path, bs )

    logger = open(os.path.join(trg_folder, log_file), 'w')
    custom_print(sys.argv)
    custom_print('\n')


    model_id = args.model_id
    if (model_id == 1):
        max_word_len = args.max_word_len
        conv_filter_size = args.cnn_filter_size

    model = ins.get_model(model_id)
    model = model.to('cuda')
    
    use_cuda = torch.cuda.is_available()
    
    random_seed(seed_value, use_cuda)

    
    optimizer, scheduler = ins.get_optimizer_scheduler(model, ins.preprocessor.train_dataloader)

    ins.train(model, optimizer, scheduler)

    custom_print('Evauating the model with the best Val Accuracy........')

    best_model = ins.load_model('best_model.pt')
    evalP, evalR, evalF = ins.evaluate_classifier(ins.preprocessor.eval_dataloader, best_model, ins.preprocessor.eval_dataset_raw, ins.preprocessor.tokenizer, eval_data_path)
    custom_print(evalP, evalR, evalF)

    custom_print("All Done :)")
    logger.close()


    

