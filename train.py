import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import NavigationLSTM

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    encode_data
)

def process_data_to_csv(filePath):
    file = open(filePath)
    train_output = open("train.csv", 'w')
    
    train_output.write("sentence|action|target")
    
    data_json = json.load(file)
    train = data_json["train"]
    for t in train:
        for i in t:
            sentence = preprocess_string(i[0])
            action = preprocess_string(i[1][0])
            target = preprocess_string(i[1][1])
            train_output.write("\n"+sentence+"|"+action+"|"+target)
    train_output.close()
    
    valid_output = open("validation.csv", 'w')
    valid_output.write("sentence|action|target")

    valid = data_json["valid_seen"]
    for v in valid:
        for i in v:
            sentence = preprocess_string(i[0])
            action = preprocess_string(i[1][0])
            target = preprocess_string(i[1][1])
            valid_output.write("\n"+sentence+"|"+action+"|"+target)
    valid_output.close()


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    
    data = pd.read_csv("train.csv", delimiter="|")
    
    #  1- lower the case
    #  2- lemmatize
    #  3- remove stop words (test)

    # sentences to lower case
    data['sentence'] = data.sentence.str.lower()
    # remove duplicates
    data = data.groupby(['sentence', 'action', 'target'], as_index=False).count()[['sentence', 'action', 'target']]

    train, valid = train_test_split(data, stratify=data[['action', 'target']], test_size=0.2, random_state=10)

    v2i, i2v, seq_len = build_tokenizer_table(train['sentence'])
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train[['action', 'target']])
    
    # test_data = pd.read_csv("validation.csv", delimiter="|")
    # t_actions_to_index, index_to_actions, t_targets_to_index, index_to_targets = build_output_tables(test_data[['action', 'target']])


    x_train,y_train = encode_data(train[['sentence', 'action']], v2i, seq_len, actions_to_index)
    x_valid,y_valid = encode_data(valid[['sentence', 'action']], v2i, seq_len, actions_to_index)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader, len(v2i), len(targets_to_index), seq_len


def setup_model(params):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = NavigationLSTM(params)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """

    # action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    return target_criterion, optimizer


def train_epoch(args,model,loader,optimizer,target_criterion,device,training=True):
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    target_preds = []
    target_labels = []
    
    # iterate over each batch in the dataloader
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        targets_out = model(inputs)
        target_loss = target_criterion(targets_out.squeeze(), labels[:].long())
        loss = target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        target_preds.extend(target_preds_.cpu().numpy())
        target_labels.extend(labels[:].cpu().numpy())

    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_target_loss, target_acc


def validate(args, model, loader, optimizer, target_criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_target_loss, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            target_criterion,
            device,
            training=False,
        )

    return val_target_loss, target_acc


def train(args, model, loaders, optimizer, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_target_loss, train_target_acc = train_epoch(args,model,loaders["train"],optimizer,target_criterion,device)

        # some logging
        print(
            # f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
            f"train target loss: {train_target_loss}"
        )
        print(
            # f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
            f"train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_target_loss, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                target_criterion,
                device,
            )

            print(
                f"\nval target loss: {val_target_loss}"
            )
            print(
                f"\nval target losaccs: {val_target_acc}"
            )

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #


def main(args):

    # process_data_to_csv("./lang_to_sem_data.json")

    device = torch.device("cpu")
    
    # get dataloaders
    train_loader, val_loader, vocab_size, output_len, seq_len = setup_dataloader(args)

    loaders = {"train": train_loader, "val": val_loader}

    params = {"seq_len":seq_len,
    "vocab_size":vocab_size, 
    "embedding_dim":100, 
    "lstm_hidden_dim":256, 
    "dropout":0.33, 
    "linear_output_dim":100, 
    "batch_size":args.batch_size,
    "number_of_outputs":output_len}

    # build model
    model = setup_model(params)
    print(model)

    # get optimizer and loss functions
    # action_criterion, target_criterion, optimizer = setup_optimizer(args, model)
    target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            # action_criterion,
            target_criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, target_criterion, device)
        # train(args, model, loaders, optimizer, action_criterion, target_criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument("--model_output_dir", type=str, help="where to save model outputs")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each batch in loader")
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs", type=int)
    parser.add_argument("--val_every", default=5, help="number of epochs between every eval loop", type=int)

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--learning_rate", default=0.001, help="learning rate", type=float)

    args = parser.parse_args()
    main(args)
