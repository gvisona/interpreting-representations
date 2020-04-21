import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from scipy.cluster.hierarchy import dendrogram
from bokeh.palettes import Category20, d3, Paired12, Viridis10, Category20c



def plot_rectangles(ax, categories, palette=Category20[20], color_idxs=None):
    """
    Fill background areas for pseudo-parallel coordinates plot
    categories must be format {"categories1_label":n_features1, ...}
    """
    ylims = ax.get_ylim()
    
    rectangles = []
    x = -0.5
    for category, n_features in categories.items():
        rect = Rectangle((x, ylims[0]), n_features, ylims[1] - ylims[0])
        rectangles.append(rect)
        ax.text(x+n_features/2, ylims[1]+0.1, category, rotation=20, fontsize=20)
        x += n_features

    facecolor=palette[:len(rectangles)] if color_idxs is None else [palette[i] for i in color_idxs]
    pc = PatchCollection(rectangles, facecolor=facecolor, alpha=0.2,
                     edgecolor=None)
    
    # handles = []
    # for col, category in zip(facecolor, categories.keys()):
    #     handles.append(Polygon([(0,0),(10,0),(0,-10)],color=col, alpha=0.2,
    #                         label=category))
    
    # ax.legend(handles=handles)
    ax.add_collection(pc)

######################################################
# Plotting a heatmap matrix with dendrograms
def plot_matrix_dendrograms(figure, matrix, hierarchy_left=None, hierarchy_top=None, cbar_label="Pearson Correlation Coefficient", cmap=plt.cm.coolwarm):
    matrix_x = 0.09 if hierarchy_left is None else 0.3
    matrix_y = 0.1
    matrix_width = 0.9 if hierarchy_left is None else 0.6
    matrix_height =  0.83 if hierarchy_top is None else 0.6
    cbar_x = matrix_width + matrix_x + 0.01
    
    if hierarchy_left is not None:
        ax1 = figure.add_axes([0.09,matrix_y,0.2,matrix_height], frameon=False)
        Z1 = dendrogram(hierarchy_left, orientation='left')
        ax1.invert_yaxis()
        ax1.set_xticks([])
        ax1.set_yticks([])
    if hierarchy_top is not None:
        ax2 = figure.add_axes([matrix_x,0.71,matrix_width,0.2], frameon=False)
        Z2 = dendrogram(hierarchy_top)
        ax2.set_xticks([])
        ax2.set_yticks([])

    axmatrix = figure.add_axes([matrix_x,matrix_y,matrix_width,matrix_height])
    im = axmatrix.matshow(matrix, aspect='auto', origin='upper', cmap=cmap)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    axcolor = figure.add_axes([cbar_x,matrix_y,0.02,matrix_height])
    cb = plt.colorbar(im, cax=axcolor)
    cb.set_label(label=cbar_label, size=18)

# Plotting a heatmap matrix with dendrograms
######################################################
