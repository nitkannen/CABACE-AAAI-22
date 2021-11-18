import io
from collections import OrderedDict
import numpy as np
import torch

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = ( fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = (tokens[1:])
        if(len(data) > 40000):
          break
    return data

def build_vocab(all_tokens):
    
    char_v = OrderedDict()
    char_v['<PAD>'] = 0
    char_v['<UNK>'] = 1
    char_idx = 2
    for tokens in all_tokens:
        for tok in tokens:
            for c in tok:
                if c not in char_v:
                    char_v[c] = char_idx
                    char_idx += 1

    return char_v

def load_char_embedding( embed_file, vocab, word_embed_dim):
    
  embed_matrix = list()
  embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))  ## FOR PAD
  embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))  ## FOR UNK

  #for i in range(2, len(vocab)):
  for k in vocab.keys():
    if k == '<PAD>' or k == '<UNK>':
      continue
    if k in embed_file:
      vec = embed_file[k]
      embed_matrix.append(vec)
    else:
      embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

  return embed_matrix




def get_embed_matrix_and_vocab(val_dataset_raw, train_dataset_raw, tokenizer):
    """
        Entry point into the script
    """
    eng_vectors = load_vectors('wiki-news-300d-1M.vec')
    all_tokens = val_dataset_raw['input_ids'] + (train_dataset_raw['input_ids']) 
    all_ = []
    for tok in all_tokens:
        all_.append(tokenizer.convert_ids_to_tokens(tok, skip_special_tokens=True))
    char_vocab = build_vocab(all_)
    embed_matrix  = load_char_embedding(eng_vectors, char_vocab, 300)


    temp_emb = embed_matrix

    new_temp_emb = np.zeros(shape = (len(embed_matrix), 300))

    for i in range(len(temp_emb)):
  
        vec = temp_emb[i]
        if type(vec) == list:
            print('changing list to numpy array')
            new_vec = np.zeros(300)
            for j in range(len(vec)):
                new_vec[j] = float(vec[j])

            new_temp_emb[i] = new_vec
        else:
            new_temp_emb[i] = vec

    embed_matrix = new_temp_emb

    embed_matrix = torch.tensor(embed_matrix)
    embed_matrix = embed_matrix.to('cuda')

    return char_vocab, embed_matrix

