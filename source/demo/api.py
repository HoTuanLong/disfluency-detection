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
            "RM": [],
            "IM": [],
        }

        sentence = vitools.normalize_diacritics(sentence)
        sentence = underthesea.word_tokenize(sentence, format = "text")
        pred = self.model(sentence)

        # print(pred)
        for entity in pred:
            if entity["entity_group"] in output:
                output[entity["entity_group"]].append(entity["word"])
        
        for entity_group, entities in output.items():
            fixed_entities = []
            i = 0
            while i < len(entities):
                if entities[i].endswith("@@"):
                    fixed_entity = entities[i][:-2] + entities[i + 1]
                    fixed_entities.append(fixed_entity.replace("_", " "))
                    i += 2
                else:
                    fixed_entity = entities[i]
                    fixed_entities.append(fixed_entity.replace("_", " "))
                    i += 1
            output[entity_group] = fixed_entities
        
        return output

# if __name__ == '__main__':

#     with open("source/config.json", 'r') as config:
#         config = config.read()
#     config = json.loads(config)
#     save_ckp_dir = config["save_dir"]
#     model = Disfluency(ckp_dir = save_ckp_dir)
#     model.prediction(sentence = "có sân bay í lộn hãng hàng không nào có các chuyến bay từ điện biên phủ đến quảng ninh à chính xác là đến quy nhơn khởi hành trước 6 giờ 30 phút sáng không")

    