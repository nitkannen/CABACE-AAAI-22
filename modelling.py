
import torch
from transformers import BertPreTrainedModel
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, AutoModel
from torch import nn

class Simple_BERT(BertPreTrainedModel):
    def __init__(self, config, actual_model_checkpoint):
        super(Simple_BERT, self).__init__(config)
        self.bert = AutoModel.from_pretrained(actual_model_checkpoint)
        self.classifier = nn.Linear(768, 5)
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.token_dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask, labels):
        lhs = self.bert(input_ids, attn_mask).last_hidden_state
        logits = self.classifier(self.token_dropout(lhs))
        ypreds = torch.argmax(self.softmax(logits), dim=2)
        loss = self.criterion(logits.view(-1, 5), labels.view(-1))
        return ypreds, loss, None, None



class Transform_CharacterBERT(BertPreTrainedModel):
      
    def __init__(self, config, actual_model_checkpoint, char_vocab, embed_matrix, tokenizer, max_word_len, cnn_size):
        super(Transform_CharacterBERT, self).__init__(config)
        self.bert = AutoModel.from_pretrained(actual_model_checkpoint)
        self.classifier = nn.Linear(868, 5)
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.char_embeddings = CharEmbeddings(len(char_vocab), 300, embed_matrix,max_word_len, cnn_size, 0.1)
        self.token_dropout = nn.Dropout(0.1)
        self.tokenizer = tokenizer
        self.max_word_len = max_word_len
        self.conv_filter_size = cnn_size
        self.char_vocab = char_vocab
    
    def get_char_seq(self, words, max_len):

        
        char_seq = list()

        for i in range(0, self.max_word_len + self.conv_filter_size - 1):  #### CLS
            char_seq.append(self.char_vocab['<PAD>'])

        for i in range(0,  self.conv_filter_size - 1):  #### Extra Padding
            char_seq.append(self.char_vocab['<PAD>'])

        for word in words:

            for c in word[0:min(len(word), self.max_word_len)]:
                if c in self.char_vocab:
                    char_seq.append(self.char_vocab[c])
                else:
                    char_seq.append(self.char_vocab['<UNK>'])
            pad_len = self.max_word_len - len(word)
            for i in range(0, pad_len):
                char_seq.append(self.char_vocab['<PAD>'])
            for i in range(0, self.conv_filter_size - 1):
                char_seq.append(self.char_vocab['<PAD>'])

        pad_len = max_len - len(words) - 1
        for i in range(0, pad_len):
            for i in range(0, self.max_word_len + self.conv_filter_size - 1):
                char_seq.append(self.char_vocab['<PAD>'])

        return char_seq

    def get_all_char_seq_tensor(self, input_ids):

        all_char_seqs = []
        max_len = input_ids.shape[1]
        
        for i in range(len(input_ids)):
            i_id = input_ids[i]
            all_toks = self.tokenizer.convert_ids_to_tokens(i_id, skip_special_tokens = True)
        
            char_seq = self.get_char_seq(all_toks, max_len)
            all_char_seqs.append(char_seq)

        return torch.tensor(all_char_seqs)


    def forward(self, input_ids, attn_mask, labels):

        lhs = self.bert(input_ids, attn_mask).last_hidden_state
        #print('LHS', lhs.device)
        all_char_seqs = self.get_all_char_seq_tensor(input_ids)
        all_char_seqs = all_char_seqs.to('cuda')
        #print('allchar', all_char_seqs.device)
        char_token_embs = self.char_embeddings(all_char_seqs)
        char_token_embs = char_token_embs.to('cuda')
        #print('chartok', char_token_embs.device)
        lhs_plus_char = torch.cat([lhs, char_token_embs], dim =  -1)

        # print("lhs shape", lhs.shape)
        logits = self.classifier(self.token_dropout(lhs_plus_char))
        ypreds = torch.argmax(self.softmax(logits), dim=2)
        loss = self.criterion(logits.view(-1, 5), labels.view(-1))
        return ypreds, loss



class CharEmbeddings(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, embed_matrix, max_word_len, conv_filter_size,  drop_out_rate):

        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0) ### B * Pad_length * 300
        self.embeddings.weight.data.copy_((embed_matrix))
        self.embeddings.weight.requires_grad = True
        self.conv1d = nn.Conv1d(embed_dim, 100, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)
        self.dropout = nn.Dropout(drop_out_rate)


    def forward(self, char_seq):

        char_embeds = self.embeddings(char_seq) ### B * Pad_length * 300
        char_embeds = self.dropout(char_embeds) ##same
        char_embeds = char_embeds.permute(0, 2, 1) ##B*300_pad_lenght
        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds))) ##B*100*num_of_tokens
        char_feature = char_feature.permute(0, 2, 1) ##B * num_of_tokens * 100

        return char_feature