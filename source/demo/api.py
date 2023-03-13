import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

class Disfluency():
    def __init__(self, ckp_dir):
        self.model = transformers.pipeline("token-classification", ckp_dir, aggregation_strategy = "simple")
    
    def prediction(self, sentence):
        output = {
            "O": [],
            "B-RM": [],
            "I-RM": [],
            "B-IM": [],
            "I-IM": []
        }

        sentence = vitools.normalize_diacritics(sentence)
        sentence = underthesea.word_tokenize(sentence, format = "text")
        pred = self.model(sentence)
        print(pred)

if __name__ == '__main__':

    with open("source/config.json", 'r') as config:
        config = config.read()
    config = json.loads(config)
    save_ckp_dir = config["save_dir"]
    model = Disfluency(ckp_dir = save_ckp_dir)
    model.prediction(sentence = "chiều à không sáng thứ hai tôi muốn bay từ cà mau đến thanh hoá")

    