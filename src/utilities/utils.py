import numpy as np
import matplotlib.pyplot as plt
import torch


# To calculate accuracy,recall and precision given the confusion matrix
def acc(c):
    return ((c[0][0] + c[1][1])/(c.sum()))

def recall(c):
    return ((c[1][1])/(c[1][0] + c[1][1]))

def precision(c):
    return ((c[1][1])/(c[1][1] + c[0][1]))


# Fn to read and parse a log file. Returns a disctionary containing the relevant metrics
def read_log_file(log_file_path , metric="acc"):

    log_file = open(log_file_path , mode="r")

    epochs = []
    loss = []
    train = []
    val = []
    test = []


    for log in log_file:

        elements = log.split()

        epochs.append(int(elements[4]))
        loss.append(float(elements[6]))

        
        c_train = np.array([[float(elements[8]) , float(elements[9])],
                            [float(elements[10]) , float(elements[11])]])
        c_val = np.array([[float(elements[13]) , float(elements[14])],
                            [float(elements[15]) , float(elements[16])]])
        c_test = np.array([[float(elements[18]) , float(elements[19])],
                            [float(elements[20]) , float(elements[21])]])
                
        if(metric == "acc"):
            train.append(acc(c_train))
            val.append(acc(c_val))
            test.append(acc(c_test))

        elif(metric == "recall"):
            train.append(recall(c_train))
            val.append(recall(c_val))
            test.append(recall(c_test))

        elif(metric == "precision"):
            train.append(precision(c_train))
            val.append(precision(c_val))
            test.append(precision(c_test))

    log_file.close()

    return {
        "epochs":epochs,
        "loss":loss,
        "train":train,
        "val":val,
        "test":test
    }


# Make a plot by specifying the parameters
def plot_metrics(x , x_axis_label , y_list , y_labels , y_axis_label , colour_list , title , linewidth=1.5):

    plt.style.use('seaborn-v0_8-darkgrid')

    for i , y in enumerate(y_list):

        plt.plot(x , y , label=y_labels[i] , color=colour_list[i] , linestyle = '-' , linewidth=linewidth )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# Helper function for dataloader to sort dict values by the keys
def get_values_sorted_by_keys(d):

    keys = list(d.keys())
    keys.sort()
    sorted_vals = np.array([ d[key] for key in keys])

    return sorted_vals


# save model at specified path

def save_model(model , path):

    torch.save(model.state_dict(), path)
