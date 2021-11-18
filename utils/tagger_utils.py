
def get_char_token_idx(sample, tokens):

    char_to_token_index = [-1 for i in range(len(sample['text']))]
    cur_ptr = 0
    token_idx = 0

    while sample['text'][cur_ptr] == ' ' and cur_ptr < len(char_to_token_index):
        cur_ptr += 1
    
    while cur_ptr < len(char_to_token_index):

        if sample['text'][cur_ptr] != ' ':
            char_offset = len(tokens[token_idx])
            if tokens[token_idx][:2] == '##':
                char_offset -= 2
            for tok_span in range(cur_ptr, cur_ptr + char_offset):
                char_to_token_index[tok_span] = token_idx
            token_idx += 1
            cur_ptr += char_offset

        while cur_ptr < len(char_to_token_index) and sample['text'][cur_ptr] == ' ':
            cur_ptr += 1

    return char_to_token_index

def token_to_span_map(tokens, char_to_token_index):

    token_to_span_map = [[0, 0] for idx in range(len(tokens))]

    for i in range(len(char_to_token_index) - 1):
        if char_to_token_index[i] != char_to_token_index[i + 1]:
            if char_to_token_index[i] >= 0:
                token_to_span_map[char_to_token_index[i]][1] = i + 1

            if char_to_token_index[i + 1] >= 0:
                token_to_span_map[char_to_token_index[i+1]][0] = i + 1

    return token_to_span_map