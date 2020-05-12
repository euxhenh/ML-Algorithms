import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


COLORS = [
    '#cc5151', '#51cccc', '#337f7f', '#8ecc51', '#7f3333', '#597f33', '#8e51cc',
    '#59337f', '#ccad51', '#7f6c33', '#51cc70', '#337f46', '#5170cc', '#33467f',
    '#cc51ad', '#7f336c', '#cc7f51', '#7f4f33', '#bccc51', '#757f33', '#60cc51',
    '#3c7f33', '#51cc9e', '#337f62', '#519ecc', '#33627f', '#6051cc', '#3c337f'
]


def scatter_points(x1, x2, labels=None, unq_labels=None, path=None):
    if labels is not None:
        unq = len(np.unique(labels))
        pal = COLORS[:unq]

        sns.scatterplot(x=x1, y=x2, hue=labels, palette=pal, linewidth=0,
                        s=10, legend='full')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
                         borderaxespad=0, labels=unq_labels)
        sns.despine(left=True, bottom=True)

        if path is not None:
            plt.savefig(path)

        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        plt.scatter(x1, x2, s=1)
        plt.xticks([])
        plt.yticks([])
        plt.show()
