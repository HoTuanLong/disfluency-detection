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

        print(sentence)
        sentence = vitools.normalize_diacritics(sentence)
        print(sentence)
        sentence = underthesea.word_tokenize(sentence, format = "text")
        print(sentence)
        pred = self.model(sentence)

        print(pred)

if __name__ == '__main__':
    
    model = Disfluency(ckp_dir = "/home/ubuntu/long.ht/disfluency/ckps/Disfluency/word/")
    model.prediction(sentence = "chiều à không sáng thứ hai tôi muốn bay từ cà mau đến thanh hoá")

    