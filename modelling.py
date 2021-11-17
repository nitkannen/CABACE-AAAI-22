
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
      
    def __init__(self, config, actual_model_checkpoint):
        super(Transform_CharacterBERT, self).__init__(config)
        self.bert = AutoModel.from_pretrained(actual_model_checkpoint)
        self.classifier = nn.Linear(868, 5)
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.char_embeddings = CharEmbeddings(len(char_vocab), 300, 0.2)
        self.token_dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask, labels):

        lhs = self.bert(input_ids, attn_mask).last_hidden_state
        #print('LHS', lhs.device)
        all_char_seqs = get_all_char_seq_tensor(input_ids)
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
    
    def __init__(self, vocab_size, embed_dim, drop_out_rate):

        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0) ### B * Pad_length * 300
        self.embeddings.weight.data.copy_((embed_matrix))
        self.embeddings.weight.requires_grad = True
        self.conv1d = nn.Conv1d(embed_dim, 100, 3)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)
        self.dropout = nn.Dropout(drop_out_rate)


    def forward(self, char_seq):

        char_embeds = self.embeddings(char_seq) ### B * Pad_length * 300
        char_embeds = self.dropout(char_embeds) ##same
        char_embeds = char_embeds.permute(0, 2, 1) ##B*300_pad_lenght
        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds))) ##B*100*num_of_tokens
        char_feature = char_feature.permute(0, 2, 1) ##B * num_of_tokens * 100

        return char_feature