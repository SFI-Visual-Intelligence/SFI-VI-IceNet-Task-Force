import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def get_filenames():
    filenames = os.listdir()
    filenames = filter(lambda f: f.startswith("history"), filenames)
    return filenames


def plot(header, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames()):
        df = pd.read_json(filename)
        df.plot(y=header, ax=ax, label=i)
        print(df.columns)
    plt.title(header)
    return ax
    # plt.savefig("figures/" + header + ".png")


def simple_plot():
    ## Loss
    fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames()):
        df = pd.read_json(filename)
        # df.plot(y=["loss", "val_loss"])
        df.plot(y="loss", ax=ax, label=i)
    plt.title("Loss")
    plt.savefig("figures/loss.png")

    ## train avg accuracy
    fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames()):
        df = pd.read_json(filename)
        df.plot(y="acc_mean", ax=ax, label=i)
    plt.title("train accuracy (mean)")
    plt.savefig("figures/train_accuracy_mean.png")

    ## val avg accuracy
    fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames()):
        df = pd.read_json(filename)
        df.plot(y="val_acc_mean", ax=ax, label=i)
    plt.title("validation accuracy (mean)")
    plt.savefig("figures/val_accuracy_mean.png")


if __name__ == "__main__":
    # simple_plot()
    ## options: ['loss', 'acc_mean', 'acc_1month', 'acc_2month', 'acc_3month',
    #   'acc_4month', 'acc_5month', 'acc_6month', 'val_loss', 'val_acc_mean',
    #   'val_acc_1month', 'val_acc_2month', 'val_acc_3month', 'val_acc_4month',
    #   'val_acc_5month', 'val_acc_6month', 'lr']
    # plot("loss")
    ax = plot("val_acc_1month")
    ax = plot("val_acc_6month", ax=ax)
    plt.show()
    ax.plot("loss")
    plt.show()
    
