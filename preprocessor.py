from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, AutoModel
from ast import literal_eval
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
import pandas as pd
import json
from torch.utils.data import DataLoader

class Preprocessor():

    def __init__(self, tokenizer_checkpoint, train_data_path, eval_data_path, batch_size):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.config = AutoConfig.from_pretrained(tokenizer_checkpoint) 
        self.label_to_index = self.return_label2id()
        self.data_collator = DataCollatorForTokenClassification(
                            self.tokenizer, pad_to_multiple_of=(8)
                        )
        self.batch_size = batch_size
        self.train_dataloader = self.read_data(train_data_path, True)
        self.eval_dataloader = self.read_data(eval_data_path, False)
        self.train_dataset_raw = self.return_dataset(train_data_path)['train']
        self.eval_dataset_raw = self.return_dataset(eval_data_path)['train']

    def return_dataset(self,datapath):

        with open(datapath, 'r') as f1: 
            data = json.load(f1)
        for val in data:
            del(val['ID'])

        df_raw = pd.DataFrame(data)
        df_raw['text'] = df_raw['text'].apply(lambda x: x.replace("’", "'").replace("’", "'"))

        df_raw.to_csv('train.csv', index = False)

        dataset = load_dataset('csv', data_files={ 'train':'train.csv'})
        #bert_dataset = dataset.map(self.create_BERT_inputs, batched = True)
        return dataset

    def read_data(self,datapath, is_train = False):

        with open(datapath, 'r') as f1: 
            data = json.load(f1)
        for val in data:
            del(val['ID'])
        df_raw = pd.DataFrame(data)
        df_raw['text'] = df_raw['text'].apply(lambda x: x.replace("’", "'").replace("’", "'"))

        df_raw.to_csv('train.csv', index = False)

        dataset = load_dataset('csv', data_files={ 'train':'train.csv'})
        bert_dataset = dataset.map(self.create_BERT_inputs, batched = True)

        return self.get_dataloader(is_train, bert_dataset)



    def get_dataloader(self,is_train, bert_dataset):

        if is_train:
            return DataLoader(
                    bert_dataset['train'], collate_fn = self.data_collator ,shuffle = True,  batch_size=self.batch_size
                    )
        
        else:
            return  DataLoader(
                    bert_dataset['train'], collate_fn = self.data_collator, batch_size=self.batch_size
                    )

    def return_label2id(self):

        label_to_index = {}
        label_to_index['O'] = 0
        label_to_index['B-SHORT'] = 1
        label_to_index['I-SHORT'] = 2
        label_to_index['B-LONG'] = 3
        label_to_index['I-LONG'] = 4 
        return label_to_index 

    def fill_acronym_tags(self,acronyms,text,target, tokens):
          
        acronyms_span_list = literal_eval(acronyms)

        for acronym_span in acronyms_span_list:
            acronym = text[acronym_span[0]: acronym_span[1]]
            sub_acronyms = self.tokenizer.tokenize(acronym)
            if len(sub_acronyms) == 0:
                continue
            for idx in range(len(tokens) + 1 - len(sub_acronyms)):
                start_token = tokens[idx]
                match = True

                if sub_acronyms[0] == start_token:
                    for j in range(idx, idx + len(sub_acronyms)):
                        if sub_acronyms[j - idx] != tokens[j]:
                            match = False
                    if match:
                        target[idx] = 'B-SHORT'
                        for k in range(idx + 1, idx + len(sub_acronyms)):
                            target[k] = 'I-SHORT'

        return target

    def fill_long_form_tags(self,long_forms,text, target, tokens):
        
        long_forms_list = literal_eval(long_forms)

        for long_form_span in long_forms_list:
            long_form = text[long_form_span[0]: long_form_span[1]]
            sub_long_forms = self.tokenizer.tokenize(long_form)
            if len(sub_long_forms) == 0:
                continue
            for idx in range(len(tokens) + 1 - len(sub_long_forms)):
                start_token = tokens[idx]
                match = True
                if sub_long_forms[0] == start_token:
                    for j in range(idx, idx + len(sub_long_forms)):
                        if sub_long_forms[j - idx] != tokens[j]:
                            match = False
                    if match:
                        target[idx] = 'B-LONG'
                        for k in range(idx + 1, idx + len(sub_long_forms)):
                            target[k] = 'I-LONG'

        return target

    def create_BERT_inputs(self,example):
    
        new_dict = {}

        tokenized_input = self.tokenizer(example['text'])
        labels = []
        for i in range(len(tokenized_input['input_ids'])):
        #print(example['text'])
            target = ['O' for k in range(len(tokenized_input['input_ids'][i]) - 2)]
            target = self.fill_acronym_tags(example['acronyms'][i], example['text'][i], target, self.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][i][1:-1]))
            target = self.fill_long_form_tags(example['long-forms'][i], example['text'][i], target, self.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][i][1:-1]))
            target = [self.label_to_index[i] for i in target]
            target = [-100] + target
            target.append(-100)
            target = target[:512]
            labels.append(target)

            tokenized_input['input_ids'][i] = tokenized_input['input_ids'][i][:512]
            tokenized_input['attention_mask'][i] = tokenized_input['attention_mask'][i][:512]
            tokenized_input['token_type_ids'][i] = tokenized_input['token_type_ids'][i][:512]

        tokenized_input['valid_token_len'] = [[len(tokenized_input['input_ids'][i]) - 2] for i in range(len(tokenized_input['input_ids']))]

        del(example['acronyms'])
        del(example['long-forms'])
        del(example['text'])
        new_dict['labels'] = labels
        new_dict.update(tokenized_input)
        #print(new_dict)
        return new_dict

    

