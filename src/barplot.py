import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

stats_dir = 'stats'

# https://www.datascience.com/blog/learn-data-science-intro-to-data-visualization-in-matplotlib
# Define a function for a bar plot
def barplot(x_data, y_data, error_data, x_label, y_label, title):
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color = '#539caf', align = 'center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
    ax.errorbar(x_data, y_data, yerr = error_data, color = '#297083', ls = 'none', lw = 2, capthick = 1, capsize=7)
    ax.set_ylabel(y_label)
    ax.set_ylim(0.6,.68)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    plt.show()

models = []
means = []
stds = []

models.append('fixed_action')
means.append(0.611794500723589)
stds.append(-float('inf'))

models.append('lin_reg')
means.append(0.6536707706338349)
stds.append(-float('inf'))


for model in ['lin_ucb', 'lasso', 'mwu_lasso', 'thompson', 'mwu_thompson']:

    accs = []
    
    for line in open(os.path.join(stats_dir, model+'_stats.txt')):
        acc = float(line.strip().split()[1])
        accs.append(acc)

    models.append(model)
    mean = np.mean(accs)
    means.append(mean)

    # replace np.std(accs)
    confidence = .95
    n = len(accs)
    se = scipy.stats.sem(accs)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print(model, mean-h, mean, mean+h)
    stds.append(h)

barplot(x_data = models
        , y_data = means
        , error_data = stds
        , x_label = 'Model'
        , y_label = 'Accuracy'
        , title = 'Model Comparison')
