import matplotlib.pyplot as plt
import numpy as np

def draw_chart(X, y, title='', txt='', you=None):
    unique = np.unique(y)
    markers = ['o', 'x']
    plt.title(title)
    plt.annotate(txt, xy=(0.01, 0.01), xycoords='axes fraction')
    colors = [
        plt.cm.jet(i / float(len(unique) - 1)) for i in range(len(unique))
    ]

    colors = [np.asarray(c).reshape((1, -1)) for c in colors]

    for i, u in enumerate(unique):
        x0 = [X[j][0] for j in range(len(y)) if y[j] == u]
        x1 = [X[j][1] for j in range(len(y)) if y[j] == u]
        plt.scatter(x0, x1, c=colors[i], s=80, label=str(u), marker=markers[i % len(markers)])


    if you is not None:
        plt.scatter([you[0][0]], [you[0][1]],
                    c=colors[0],
                    s=100,
                    label='you',
                    marker='*')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
