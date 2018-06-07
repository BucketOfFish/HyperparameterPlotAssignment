# Make plots from hyperparameter scans

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itr

files = "Data/*"
files = glob.glob(files)

#parameter_space = [[3, 4, 5, 6, 7], [64, 128, 256, 512, 1024], [0.0001, 0.0005, 0.001]
#                   ,[0.2, 0.3, 0.4, 0.5], [11, 21, 31, 41, 51]]
parameter_space = [[3], [64, 128, 256, 512], [0.0001, 0.0005, 0.001] # <-- erase this
                   ,[0.2, 0.3, 0.4, 0.5], [11, 21, 31, 41, 51]] # <-- erase this
#parameter_shape = [len(dimension) for dimension in parameter_space]
axlabels = ["Hidden Layers", "Neurons per Hidden Layer", "Learning Rate",
            "Dropout Probability", "Window Size"]
parameter_shape = [1, 4, 3, 4, 5] # <-- erase this line when you have all the data
#correct_settings = (1, 2, 3, 4)
correct_settings = (0, 2, 2, 3, 4) # <-- erase this line when you have all the data
n_parameters = len(parameter_shape)
losses = np.empty(parameter_shape)
accuracies = np.empty(parameter_shape)


for file_name in files:
    # get classifier test loss and accuracy
    file = open(file_name)
    lines = file.readlines()
    file.close()
    results = lines[-2].rstrip().replace(';', '')
    results = results.split(" ")
    indices = [i for i, x in enumerate(results) if x == '(C)']
    test_loss = results[indices[0]+1]
    test_accuracy = results[indices[1]+1]
    # parse filename
    file_name = file_name.split("/")[-1]
    file_name = file_name.split(".log")[0]
    parameters = file_name.split("_")[1:-1]
    parameters = [float(i) for i in parameters]
    # store values
    p = [parameter_space[i].index(x) for i, x in enumerate(parameters)]
    losses[p[0]][p[1]][p[2]][p[3]][p[4]] = test_loss
    accuracies[p[0]][p[1]][p[2]][p[3]][p[4]] = test_accuracy





for k in range(5):
    lossesy = np.empty(parameter_shape[k])
    accuracyy = np.empty(parameter_shape[k])
    index = correct_settings
    for i in range(parameter_shape[k]):
        index = list(index)
        index[k] = i
        index = tuple(index)
        lossesy[i] = losses[index]
        accuracyy[i] = accuracies[index]
    plt.plot(parameter_space[k],accuracyy, 'b+-', linewidth = 1, markersize=18)
    plt.xlabel(axlabels[k])
    plt.ylabel("accuracy")
    plt.title("Accuracy vs " + axlabels[k])
#    plt.show()
    plt.plot(parameter_space[k],lossesy, 'r+-', linewidth = 1, markersize = 18)
    plt.xlabel(axlabels[k])
    plt.ylabel("loss")
    plt.title("Loss vs " + axlabels[k])
#    plt.show()

combinations = list(itr.combinations(range(5), 2))
for k in combinations:
    x = k[0]
    y = k[1]
    lossesx = np.empty([parameter_shape[x],parameter_shape[y]])
    accuracyx = np.empty([parameter_shape[x],parameter_shape[y]])
    for i in range(parameter_shape[x]):
        for j in range(parameter_shape[y]):
            index = list(correct_settings)
            index[x] = i
            index[y] = j
            index = tuple(index)
            lossesx[i][j] = losses[index]
            accuracyx[i][j] = accuracies[index]
    acc = pd.DataFrame(accuracyx, index = parameter_space[x], columns = parameter_space[y])
    loss = pd.DataFrame(lossesx, index = parameter_space[x], columns = parameter_space[y])
    ax = sns.heatmap(acc, annot = True, fmt = "")
    plt.xlabel(axlabels[y])
    plt.ylabel(axlabels[x])
    plt.title("Accuracy")
#    plt.show()
    ax = sns.heatmap(loss, annot = True, fmt = "")
    plt.xlabel(axlabels[y])
    plt.ylabel(axlabels[x])
    plt.title("Loss")
#    plt.show()
