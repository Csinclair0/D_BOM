import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


#plot noise as gray
def plot_4D(X,  title, ax):
    """create plot that includes a 3D plot and label clusters

    Parameters
    ----------
    X : dataframe
        clustered dataframe
    iterator : type
        Description of parameter `iterator`.
    title : type
        Description of parameter `title`.
    ax : type
        Description of parameter `ax`.

    Returns
    -------
    axis
        plot

    """
    colors = ['blue', 'green', 'yellow', 'cyan', 'magenta']
    for i, d in enumerate(np.unique(X.cluster)):
        P = X[X.cluster == d].values
        c = colors[d%len(colors)]
        if d >= 0:
            ax.scatter(P[:, 0], P[:, 1],P[:, 2], c= c ,  alpha = .25)
        elif d == -2:
            ax.scatter(P[:, 0], P[:, 1],P[:, 2], c = 'red' ,  alpha = .5)
        else:
            ax.scatter(P[:, 0], P[:, 1],P[:, 2], c= 'gray' ,  alpha = .05)
    ax.set_title(title)
    return ax



def plot_reach(clust):
    """reachability plot

    Parameters
    ----------
    clust : object
        Optics cluster class

    Returns
    -------
    type
        Description of returned object.

    """
    space = np.arange(len(clust.labels_))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(111)
    # Reachability plot
    colors = ['b.', 'g.', 'y.', 'c.', 'm.']
    for k in range(0, len([x for x in np.unique(clust.labels_) if x >=0])):
        col = colors[k%len(colors)]
        Xk = space[labels == k]
        Rk = reachability[labels == k]
        ax1.plot(Xk, Rk, col, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space[labels == -2], reachability[labels == -1], 'r.', alpha=.75)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')


    plt.tight_layout()
    plt.show()


def print_table(image_loc, table):
    """Deprecated, prints table to image

    Parameters
    ----------
    image_loc : location to save filter
    table : table to print

    Returns
    -------
    saved png

    """

    plt.figure(figsize = (5,5))
    plt.subplot(121)

    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    plt.table(cellText=cell_text, colLabels=table.columns, loc='center', fontsize = 100000, cellLoc = 'center')
    plt.axis('off')

    plt.savefig(image_loc + 'skuscores.png')
    plt.clf()
    plt.cla()
    plt.close()
