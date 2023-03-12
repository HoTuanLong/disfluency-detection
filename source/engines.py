from libs import *
from metrics import *

def train(train_loaders, num_epochs, model, optimizer, save_ckp_dir="./"):
    print("\nStart Training " + "-"*16)

    model = model.cuda()
    best_f1 = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"{epoch}/{num_epochs}\n" + "-"*16)

        model.train()
        running_loss = 0.0
        running_tags, running_preds = [], []

        for words, tags in tqdm.tqdm(train_loaders["train"]):
            words, tags = words.cuda(), tags.cuda()
            masks = (words != train_loaders["train"].dataset.tokenizer.pad_token_id).type(words.type()).cuda()

            logits = model(words, masks).logits.view(-1, len(train_loaders["train"].dataset.tag_names))
            loss = F.cross_entropy(logits, tags.long().view(-1))

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss += loss.item()*words.size(0)

            running_tags.extend(list(tags.long().view(-1).detach().cpu().numpy()))
            running_preds.extend(list(np.argmax(logits.detach().cpu().numpy(), axis=1)))

        train_loss, train_f1 = running_loss/len(train_loaders["train"].dataset), f1_score(np.array(running_tags), np.array(running_preds), 
                                                                                            tag_names=train_loaders["train"].dataset.tag_names)
        print(f"train - loss: {train_loss}, f1:{train_f1}")

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_tags, running_preds = [], []
            for words, tags in tqdm.tqdm(train_loaders["val"]):
                words, tags = words.cuda(), tags.cuda()
                masks = (words != train_loaders["val"].dataset.tokenizer.pad_token_id).type(words.type()).cuda()

                logits = model(words, masks).logits.view(-1, len(train_loaders["val"].dataset.tag_names))
                loss = F.cross_entropy(logits, tags.long().view(-1))

                running_loss += loss.item()*words.size(0)

                running_tags.extend(list(tags.long().view(-1).detach().cpu().numpy()))
                running_preds.extend(list(np.argmax(logits.detach().cpu().numpy(), axis=1)))
        
        val_loss, val_f1 = running_loss/len(train_loaders["val"]), f1_score(np.array(running_tags), np.array(running_preds), 
                                                                                            tag_names=train_loaders["val"].dataset.tag_names)
        if val_f1 > best_f1:
            torch.save(model, f"{save_ckp_dir}/best.ptl")
            best_f1 = val_f1
    
    print("\Finish Training " + "-"*16)

def test(test_loader, model):
    print("\n Start Testing" + "-"*16)

    model = model.cuda()

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_tags, running_preds = [], []

        for words, tags in tqdm.tqdm(test_loader):
            words, tags = words.cuda(), tags.cuda()
            masks = (words != test_loader.dataset.tokenizer.pad_token_id).type(words.type()).cuda()

            logits = model(words, masks).logits.view(-1, len(test_loader.dataset.tag_names))
            loss = F.cross_entropy(logits, tags.long().view(-1))

            running_loss += loss.item()*words.size(0)
            running_tags.extend(list(tags.long().view(-1).detach().cpu().numpy()))
            running_preds.extend(list(np.argmax(logits.detach().cpu().numpy(), axis = 1)))
    
    test_report = classification_report(
        np.array(running_tags), np.array(running_preds)
        , tag_names = test_loader.dataset.tag_names
    )
    print("Test Report:\n")
    print(test_report)

    print("\n Finish Testing" + "-"*16)
