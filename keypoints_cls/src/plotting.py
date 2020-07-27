import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, nb_epoch, log_dir):
    lossNames = ["losses", "kpt_losses", "cls_losses"]
    
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
    # loop over the loss names
    for (i, l) in enumerate(lossNames):
    	# plot the loss for both the training and validation data
    	title = "Loss for {}".format(l) if l != "losses" else "Total loss"
    	ax[i].set_title(title)
    	ax[i].set_xlabel("Epoch #")
    	ax[i].set_ylabel("Loss")
    	ax[i].plot(np.arange(0, nb_epoch), history["train_" + l], label=l)
    	ax[i].plot(np.arange(0, nb_epoch), history["test_" + l],
    		label="test_" + l)
    	ax[i].legend()
    # save the losses figure
    plt.tight_layout()
    plt.savefig("{}/losses.png".format(log_dir))
    plt.close()

    accuracyName = "cls_accs"
    plt.style.use("ggplot")
    # loop over the accuracy names
    # plot the loss for both the training and validation data
    plt.title("Accuracy for {}".format(accuracyName))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(0, nb_epoch), history["train_" + accuracyName], label=accuracyName)
    plt.plot(np.arange(0, nb_epoch), history["test_" + accuracyName], \
           label="test_" + accuracyName)
    plt.legend()
    # save the accuracies figure
    plt.tight_layout()
    plt.savefig("{}/accs.png".format(log_dir))
    plt.close()
