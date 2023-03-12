from libs import *
import keras.utils as utils

def capitalized_string(string):
    split_string = string.split("_")
    capitalized = [vitools.normalize_diacritics(word) for word in split_string]
    capitalized_string = "_".join(capitalized)
    return capitalized_string

def pad_sequence(sequences, value):
    sequence = utils.pad_sequences([sequences], 64, truncating="post", padding="post", value=value)[0].tolist()
    return sequence

def encode_ner(sequences, tags, tokenizer, tag_names):
    encode_sequences, encode_tags = [], []
    for word, tag in zip(sequences, tags):
        tokens = tokenizer.tokenize(capitalized_string(word))
        encode_sequences.extend(tokenizer.convert_tokens_to_ids(tokens))
        encode_tags.extend([tag_names.index(tag)] + [-100]*(len(tokens)-1))
    encode_sequences = [tokenizer.cls_token_id] + pad_sequence(encode_sequences, tokenizer.pad_token_id) + [tokenizer.sep_token_id]
    encode_tags = [-100] + pad_sequence(encode_tags, -100) + [-100]
    return encode_sequences, encode_tags