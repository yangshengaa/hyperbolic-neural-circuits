"""mlr training"""

# load packages
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import NLLLoss


# load file
from data import load_data
from model import MLR, HMLR
from util import train_val_test_split

# force 64
torch.set_default_dtype(torch.float64)

# ================== arguments ====================
parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="mnist", help="the name of the dataset")
parser.add_argument(
    "--model",
    type=str,
    default="HAE",
    help="the type of model",
    choices=["MLR", "HMLR"],
)
parser.add_argument("--nl", type=str, default="ReLU", help="nonlinearity")
parser.add_argument(
    "--hidden-dim", type=int, default=10, help="the dimension of the latent dim"
)
parser.add_argument(
    "--num-hidden-layers", type=int, default=2, help="the number of hidden layers"
)
parser.add_argument(
    "--epochs", type=int, default=3000, help="the number of epochs to train"
)
parser.add_argument(
    "--batchsize", type=int, default=500, help="the number of samples per batch"
)
parser.add_argument(
    "--seed", type=int, default=2022, help="the seed governing train test split"
)

parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate")
parser.add_argument(
    "--test-size",
    type=float,
    default=0.25,
    help="the proportion of data in the testing set",
)
parser.add_argument("--val-size", type=float, default=0.15, help="validation size")

parser.add_argument("--freq", type=int, default=200, help="the printing frequency")

parser.add_argument(
    "--no-gpu", action="store_true", default=False, help="True to disable using gpu"
)

args = parser.parse_args()

# specify device
device = torch.device(
    "cuda" if (torch.cuda.is_available() and not args.no_gpu) else "cpu"
)

# =============== load data =============
X, y = load_data(args.data, read_labels=True)
n, p = X.shape
train_index, val_index, test_index = train_val_test_split(
    n, seed=args.seed, test_size=args.test_size, val_size=args.val_size
)

# pass to torch
X_tensor = torch.tensor(X, device=device)
y_tensor = torch.LongTensor(y, device=device)
criterion = NLLLoss()

# ============= load model ===============
def load_model():
    modelC = MLR if args.model == "MLR" else HMLR
    model = modelC(
        in_dim=p,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        out_dim=len(np.unique(y)),
        nl=getattr(nn, args.nl)(),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    return model, optimizer


# ============ driver ============
def train():
    # search
    model, optimizer = load_model()
    X_train, X_val = X_tensor[train_index], X_tensor[val_index]
    y_train, y_val = y_tensor[train_index], y_tensor[val_index]
    num_batches = int(np.ceil(len(train_index) / args.batchsize))

    # training
    model.train()
    for e in range(args.epochs + 1):
        for b in range(num_batches):
            cur_batch = X_train[b * args.batchsize : (b + 1) * args.batchsize]
            cur_batch_y = y_train[b * args.batchsize : (b + 1) * args.batchsize]
            optimizer.zero_grad()
            logits = model(cur_batch)
            loss = criterion(logits, cur_batch_y)
            loss.backward()
            optimizer.step()
            # raise Exception()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = (pred == cur_batch_y).to(torch.float32).mean()

            # if e % args.freq == 0:
            print(f"epoch: {e:3d}, train loss: {loss:.4f}, acc: {acc:.4f}")

    # validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val)
        pred = torch.argmax(val_logits, dim=1)
        acc = (pred == y_val).to(torch.float32).mean()

    # report
    print(f"validation loss: {val_loss:.4f}, acc: {acc:.4f}")

    return model, acc


def test(model):
    X_test = X_tensor[test_index]
    y_test = y_tensor[test_index]
    with torch.no_grad():
        test_logits = model(X_test)
        test_loss = criterion(test_logits, y_test)
        pred = torch.argmax(test_logits, dim=1)
        acc = (pred == y_test).to(torch.float32).mean()

    # report
    print(f"test loss: {test_loss:.4f}, acc {acc:.4f}")

    return acc


def main():
    model, val_loss = train()
    test_loss = test(model)

    # log performance
    model_info = f"{args.data}_{args.model}_{args.hidden_dim}_{args.num_hidden_layers}_{args.lr}_{args.nl}"
    with open(os.path.join("result", "mlr.txt"), "a+") as f:
        f.write(f"{model_info},{val_loss},{test_loss}\n")


if __name__ == "__main__":
    main()
