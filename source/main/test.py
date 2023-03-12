import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *
from data import *
from engines import *



if __name__ == "__main__":
    with open("source/config.json", 'r') as config:
            config = config.read()
    config = json.loads(config)

    tag_names = [
        "O",
        "B-RM",
        "I-RM",
        "B-IM",
        "I-IM"
    ]

    test_loader = torch.utils.data.DataLoader(
        Dataset(
            data_path = config["test_path"],
            tag_names = tag_names
        ),
        num_workers = config["num_workers"],
        batch_size = config["batch_size"],
        shuffle = True
    )

    model = transformers.RobertaForTokenClassification.from_pretrained(
        "vinai/phobert-large",
        num_labels = len(tag_names)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config["learning_rate"]
    )

    save_ckp_dir = config["save_dir"]
    model = torch.load(
        "{}/best.ptl".format(save_ckp_dir), 
        map_location = "cuda", 
    )
    test(
        test_loader, 
        model, 
    )