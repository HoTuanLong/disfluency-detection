import os, sys
from libs import *
from utils_fn import *

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        tag_names
    ):
        self.data = [json.loads(sample) for sample in open(data_path, encoding='utf-8')]
        self.tag_names = tag_names
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "vinai/phobert-large",
            use_fast = False,
        )

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        sample = self.data[index]
        words, tags = sample["words"], sample["tags"]
        encoded_words, encoded_tags = encode_ner(words, tags, tokenizer=self.tokenizer, tag_names=self.tag_names)
        return np.array(encoded_words), np.array(encoded_tags)