import argparse
import json
from transformers import AutoTokenizer
import os


def prep_BIO_tags(data, trg_file_path):

    with open(trg_file_path, mode="w", encoding="utf-8") as f:
        i = 0
        for sample in data:
            print(i)
            tokens = (tokenizer.tokenize(sample['text']))
            target = [0 for k in range(len(tokens))]
            target = fill_acronym_tags(sample, target, tokens)
            target = fill_long_form_tags(sample, target, tokens)
            print(sample['text'])
            f.write(sample['text']) 
            f.write('\t') 
            print(' '.join(target))
            f.write(' '.join(target))
            f.write('\n')
            i += 1
    



def fill_acronym_tags(sample,target, tokens):

    data = sample
    
    acronyms_span_list = data['acronyms']

    tokens = (tokenizer.tokenize(data['text']))
    target = ['O' for k in range(len(tokens))]
    acronyms_span_list = data['acronyms']

    for acronym_span in acronyms_span_list:
        acronym = data['text'][acronym_span[0]: acronym_span[1]]
        sub_acronyms = tokenizer.tokenize(acronym)

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


    
def fill_long_form_tags(sample, target, tokens):
    
    data = sample
    long_forms_list = data['long-forms']

    for long_form_span in long_forms_list:
        long_form = data['text'][long_form_span[0]: long_form_span[1]]
        sub_long_forms = tokenizer.tokenize(long_form)

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







if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str,
                        help='src_file_path')
    parser.add_argument('-t', type=str,
                        help='trg_file_path')



    args = parser.parse_args()
    print(os.getcwd())
    src_file_path = args.s #'AAAI-22-SDU-shared-task-1-AE/data/english/legal/dev.json' #args.s 
    trg_file_path = args.t
    with open(src_file_path, encoding="utf8") as f1: 
        data = json.load(f1)
    #print(data)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    prep_BIO_tags(data, trg_file_path)

    