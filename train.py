import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
import json
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from model import NavigationLSTM
import matplotlib.pyplot as plt

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    encode_data
)

def process_data_to_csv(filePath):
    file = open(filePath)
    train_output = open("train_raw.csv", 'w')
    
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
    
    valid_output = open("validation_raw.csv", 'w')
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
    
    data = pd.read_csv("train_raw.csv", delimiter="|")
    
    # remove duplicates
    if args.remove_dup:
        print("removing duplicates in data")
        data = data.groupby(['sentence', 'action', 'target'], as_index=False).count()[['sentence', 'action', 'target']]
    

    train = data

    v2i, i2v, seq_len = build_tokenizer_table(train['sentence'])
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train[['action', 'target']])
    
    valid = pd.read_csv("validation_raw.csv", delimiter="|")

    x_train, y_action_train = encode_data(train[['sentence','action']], v2i, seq_len, actions_to_index)
    x_valid, y_action_valid = encode_data(valid[['sentence','action']], v2i, seq_len, actions_to_index)

    _, y_target_train = encode_data(train[['sentence','target']], v2i, seq_len, targets_to_index)
    _, y_target_valid = encode_data(valid[['sentence','target']], v2i, seq_len, targets_to_index)

    lables_train = np.array((y_action_train, y_target_train)).T
    lables_valid = np.array((y_action_valid, y_target_valid)).T

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(lables_train))
    val_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(lables_valid))
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader, len(v2i), len(actions_to_index), len(targets_to_index), seq_len


def setup_model(params):
    model = NavigationLSTM(params)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """

    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return action_criterion, target_criterion, optimizer


def train_epoch(args,model,loader,optimizer,action_criterion, target_criterion,device,training=True):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0    

    # keep track of the model predictions for computing accuracy
    target_preds = []
    target_labels = []
    
    action_preds = []
    action_labels = []

    for (inputs, labels) in tqdm.tqdm(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        actions_out, targets_out = model(inputs)

        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        
        target_preds.extend(target_preds_.cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(args, model, loader, optimizer, action_criterion, target_criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    # loss
    train_target_loss_list = []
    val_target_loss_list = []
    train_action_loss_list = []
    val_action_loss_list = []

    # accuracy
    train_target_accuracy_list = []
    val_target_accuracy_list = []
    train_action_accuracy_list = []
    val_action_accuracy_list = []

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        train_target_loss_list.append(train_target_loss)
        train_action_loss_list.append(train_action_loss)

        train_target_accuracy_list.append(train_target_acc)
        train_action_accuracy_list.append(train_action_acc)

        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )
            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target loss: {val_target_acc}"
            )
            val_target_loss_list.append(val_target_loss)
            val_action_loss_list.append(val_action_loss)

            val_target_accuracy_list.append(val_target_acc)
            val_action_accuracy_list.append(val_action_acc)

    train_epochs = range(0,args.num_epochs)
    val_epochs  = range(0,args.num_epochs,args.val_every)

    # Target and Action loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_epochs, train_target_loss_list, 'b', label='target training loss')
    ax[0].plot(val_epochs, val_target_loss_list, 'g', label='target validation loss')
    ax[0].set_title('Target Training and Validation loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(train_epochs, train_action_loss_list, 'b', label='action training loss')
    ax[1].plot(val_epochs, val_action_loss_list, 'g', label='action validation loss')
    ax[1].set_title('Action Training and Validation loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    fig.savefig(f'./plots/target_action_loss_epoch{args.num_epochs}_embed{args.embedding_dim}_dropout{args.dropout}_lstm_dim{args.lstm_hidden_dim}.png')

    # Target and Action Accuracy
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_epochs, train_target_accuracy_list, 'b', label='target training accuracy')
    ax[0].plot(val_epochs, val_target_accuracy_list, 'g', label='target validation accuracy')
    ax[0].set_title('Target Training and Validation Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[1].plot(train_epochs, train_action_accuracy_list, 'b', label='action training accuracy')
    ax[1].plot(val_epochs, val_action_accuracy_list, 'g', label='action validation accuracy')
    ax[1].set_title('Action Training and Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    fig.savefig(f'./plots/action_target_accuracy_epoch{args.num_epochs}_embed{args.embedding_dim}_dropout{args.dropout}_lstm_dim{args.lstm_hidden_dim}.png')



def main(args):

    # process_data_to_csv("./lang_to_sem_data.json")

    device = torch.device("cpu")
    
    # get dataloaders
    train_loader, val_loader, vocab_size, action_outputs, target_outputs, seq_len = setup_dataloader(args)

    loaders = {"train": train_loader, "val": val_loader}

    params = {"seq_len":seq_len,
    "vocab_size":vocab_size, 
    "embedding_dim":args.embedding_dim, 
    "lstm_hidden_dim":args.lstm_hidden_dim, 
    "dropout":args.dropout, 
    "batch_size":args.batch_size,
    "action_outputs":action_outputs,
    "target_outputs":target_outputs}

    # build model
    model = setup_model(params)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, action_criterion, target_criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each batch in loader")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs", type=int)
    parser.add_argument("--val_every", default=5, help="number of epochs between every eval loop", type=int)

    parser.add_argument("--learning_rate", default=0.001, help="learning rate", type=float)
    parser.add_argument("--embedding_dim", default=100, help="number of embedding dimensions", type=int)
    parser.add_argument("--dropout", default=0.33, help="dropout rate of the neural net", type=float)
    parser.add_argument("--lstm_hidden_dim", default=256, help="LSTM hidden dimension size", type=int)
    parser.add_argument("--weight_decay", default=0, help="L2 regularization", type=float)

    parser.add_argument("--force_cpu", default=False, action="store_true", help="debug mode")
    parser.add_argument("--remove_dup", default=False, action="store_true", help="remove duplicate records from data")
    args, unkown = parser.parse_known_args()

    main(args)
