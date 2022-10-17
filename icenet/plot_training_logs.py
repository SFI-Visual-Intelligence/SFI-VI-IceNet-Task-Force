import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path


HISTORY_DIR = "trained_networks/2021_06_15_1854_icenet_nature_communications/unet_tempscale/training_logs/dropout_20/"
# HISTORY_DIR = 


def get_filenames(folder: str):
    """
    Generator that yields absolute path for every file in the given folder that starts with 'history'.

    This function is heavily based on a very helpful stackoverflow post by wim:
    https://stackoverflow.com/questions/9816816/get-absolute-paths-of-all-files-in-a-directory
    """
    # filenames = os.listdir(folder)
    # filenames = filter(lambda f: f.startswith("history"), filenames)
    for root, dirs, files in os.walk(os.path.abspath(folder)):
        for file in files:
            if file.startswith("history"):
                yield os.path.join(root, file)


def plot(header, folder: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames(folder)):
        df = pd.read_json(filename)
        df.plot(y=header, ax=ax, label=i)
        print(df.columns)
    plt.title(header)
    return ax
    # plt.savefig("figures/" + header + ".png")


def simple_plot(folder: str, id: str = ""):
    ## Loss
    fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames(folder)):
        df = pd.read_json(filename)
        # df.plot(y=["loss", "val_loss"])
        df.plot(y="loss", ax=ax, label=i)
    plt.title("Loss")
    plt.grid("on")
    plt.savefig(f"figures/loss_{id}.png")

    ## train avg accuracy
    fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames(folder)):
        df = pd.read_json(filename)
        df.plot(y="acc_mean", ax=ax, label=i)
    plt.title("train accuracy (mean)")
    plt.grid("on")
    plt.savefig(f"figures/train_accuracy_mean_{id}.png")

    ## val avg accuracy
    fig, ax = plt.subplots()
    for i, filename in enumerate(get_filenames(folder)):
        df = pd.read_json(filename)
        df.plot(y="val_acc_mean", ax=ax, label=i)
    plt.title("validation accuracy (mean)")
    plt.grid("on")
    plt.savefig(f"figures/val_accuracy_mean_{id}.png")


def get_stats(path: str = HISTORY_DIR):
    overview = pd.DataFrame(dict(
            loss=[],
            acc_mean=[],
            acc_1month=[],
            acc_2month=[],
            acc_3month=[],
            acc_4month=[],
            acc_5month=[],
            acc_6month=[],
            val_loss=[],
            val_acc_mean=[],
            val_acc_1month=[],
            val_acc_2month=[],
            val_acc_3month=[],
            val_acc_4month=[],
            val_acc_5month=[],
            val_acc_6month=[],
            lr=[],
        )
    )
    for file in get_filenames(path):
        file_stats = pd.read_json(file).iloc[-1]  ## get only last value for each
        overview = pd.concat((overview, pd.DataFrame(file_stats).transpose()))
    return overview.describe().transpose()


if __name__ == "__main__":
    original_folder = "/Users/hjo109/Documents/GitHub/icenet-paper/trained_networks/2021_06_15_1854_icenet_nature_communications/unet_tempscale/training_logs/original_architecture"
    dropout_folder = "/Users/hjo109/Documents/GitHub/icenet-paper/trained_networks/2021_06_15_1854_icenet_nature_communications/unet_tempscale/training_logs/dropout_20/"
    
    stats_og = get_stats(original_folder)
    stats_dropout = get_stats(dropout_folder)

    print(stats_og-stats_dropout) ## All minimum values are better, but average is still a little lower.
    
    # simple_plot(Path(HISTORY_DIR).parts[-1])
    # get_stats(HISTORY_DIR)
    ## options: ['loss', 'acc_mean', 'acc_1month', 'acc_2month', 'acc_3month',
    #   'acc_4month', 'acc_5month', 'acc_6month', 'val_loss', 'val_acc_mean',
    #   'val_acc_1month', 'val_acc_2month', 'val_acc_3month', 'val_acc_4month',
    #   'val_acc_5month', 'val_acc_6month', 'lr']
    # plot("loss")
    # plot("loss")
    # plt.show()

    # ax = plot("val_acc_1month")
    # ax = plot("val_acc_6month", ax=ax)
    # plt.show()
