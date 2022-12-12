"""autoencoder training"""

# load packages
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


# load file
from data import load_data
from model import AE, HAE
from util import train_val_test_split, distortion

# force 64
torch.set_default_dtype(torch.float64)

# ================== arguments ====================
parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="mnist", help="the name of the dataset")
parser.add_argument(
    "--model", type=str, default="HAE", help="the type of model", choices=["AE", "HAE"]
)
parser.add_argument("--nl", type=str, default="ReLU", help="nonlinearity")
parser.add_argument(
    "--hidden-dim", type=int, default=10, help="the dimension of the latent dim"
)
parser.add_argument(
    "--num-hidden-layers", type=int, default=2, help="the number of hidden layers"
)
parser.add_argument("--latent-dim", type=int, default=2, help="the latent dimension")
parser.add_argument(
    "--epochs", type=int, default=3000, help="the number of epochs to train"
)
parser.add_argument(
    "--batchsize", type=int, default=500, help="the number of samples per batch"
)
parser.add_argument(
    "--seed", type=int, default=2022, help="the seed governing train test split"
)
parser.add_argument(
    "--gamma-list",
    type=float,
    nargs="+",
    default=[0, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    help="the gamma to be tested",
)

parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
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
X = load_data(args.data, read_labels=False)
n, p = X.shape
train_index, val_index, test_index = train_val_test_split(
    n, seed=args.seed, test_size=args.test_size, val_size=args.val_size
)

# pass to torch
X_tensor = torch.tensor(X, device=device)

# ============= load model ===============
def load_model():
    modelC = AE if args.model == "AE" else HAE
    model = modelC(
        in_dim=p,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        latent_dim=args.latent_dim,
        nl=getattr(nn, args.nl)(),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    return model, optimizer


# ============ driver ============
def train():
    loss_by_gamma = []
    distortion_by_gamma = []
    models = []

    # search
    for gamma in args.gamma_list:
        model, optimizer = load_model()
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        num_batches = int(np.ceil(len(train_index) / args.batchsize))

        # training
        model.train()
        for e in range(args.epochs + 1):
            for b in range(num_batches):
                cur_batch = X_train[b * args.batchsize : (b + 1) * args.batchsize]
                optimizer.zero_grad()
                X_latent, X_train_recon = model(cur_batch)
                reconstruction_loss = (
                    ((cur_batch - X_train_recon) ** 2).sum(dim=1).mean()
                )
                distortion_loss, scaled_loss = distortion(
                    cur_batch, X_latent, is_hyperbolic=args.model == "HAE"
                )
                loss = reconstruction_loss + gamma * scaled_loss
                loss.backward()
                optimizer.step()

                if e % args.freq == 0:
                    print(
                        f"epoch: {e:4d}, train loss: {loss:.4f}, distortion {distortion_loss:.4f}, scaled {scaled_loss:.4f}"
                    )

        # validation
        model.eval()
        with torch.no_grad():
            X_val_latent, X_val_recon = model(X_val)
            val_reconstruction_loss = ((X_val - X_val_recon) ** 2).sum(dim=1).mean()
            val_distortion_loss, val_scaled_loss = distortion(
                X_val, X_val_latent, is_hyperbolic=args.model == "HAE"
            )
            val_loss = val_reconstruction_loss + gamma * val_scaled_loss

        # report
        print(
            f"validation loss: {val_loss:.4f}, distortion {val_distortion_loss:.4f}, scaled {scaled_loss:.4f}"
        )

        # append
        loss_by_gamma.append(val_loss)
        distortion_by_gamma.append(val_distortion_loss)
        models.append(model)

    # find out the best
    min_idx = np.argmin(loss_by_gamma)
    opt_gamma = args.gamma_list[min_idx]
    opt_model = models[min_idx]
    opt_distortion = distortion_by_gamma[min_idx]

    return opt_gamma, opt_model, opt_distortion, min(loss_by_gamma)


def test(model):
    X_test = X_tensor[test_index]
    with torch.no_grad():
        X_test_latent, X_test_recon = model(X_test)
        test_reconstruction_loss = ((X_test - X_test_recon) ** 2).sum(dim=1).mean()
        test_distortion_loss, test_scaled_loss = distortion(
            X_test, X_test_latent, is_hyperbolic=args.model == "HAE"
        )

    # report
    print(
        f"test loss: {test_reconstruction_loss:.4f}, distortion: {test_distortion_loss:.4f}, scaled: {test_scaled_loss:.4f}"
    )

    return X_test_latent, test_reconstruction_loss, test_distortion_loss


def main():
    gamma, model, val_distortion, val_loss = train()
    test_latent, test_loss, test_distortion_loss = test(model)

    # log performance
    model_info = f"{args.data}_{args.model}_{args.hidden_dim}_{args.num_hidden_layers}_{args.latent_dim}_{args.lr}_{args.nl}_{gamma}"
    with open(os.path.join("result", "ae.txt"), "a+") as f:
        f.write(
            f"{model_info},{val_loss},{val_distortion},{test_loss},{test_distortion_loss}\n"
        )

    # log latent
    model_path = os.path.join("model", model_info)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, "model_state_dict.pt"))


if __name__ == "__main__":
    main()
