import os
from pathlib import Path
from typing import Optional, List, Union, Tuple
from itertools import product

import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes


def change_cwd_for_jupyter():
    os.chdir(Path(os.getcwd()).parent)


_AX_TYPE = object


def trim_axs(axs: Union[_AX_TYPE, np.ndarray], nb: int) -> Union[np.ndarray, _AX_TYPE]:
    """Reduce `axs` to `nb` Axes.

    All further Axes are removed from the figure.

    """
    if isinstance(axs, object):
        return axs

    axs = axs.flat  # type: ignore[union-attr]
    for ax in axs[nb:]:
        ax.remove()
    return axs[:nb]


def plot_confusion_matrix(
    confmat: pt.Tensor,
    ax: Optional[matplotlib.axes.Axes] = None,
    add_text: bool = True,
    labels: Optional[List[Union[int, str]]] = None,
) -> Tuple[object, object]:
    """Plot an confusion matrix.

    Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/confusion_matrix.py. Works for both binary, multiclass and multilabel confusion matrices.

    Args:
        confmat: the confusion matrix. Either should be an [N,N] matrix in the binary and multiclass cases or an [N, 2, 2] matrix for multilabel classification
        ax: Axis from a figure. If not provided, a new figure and axis will be created
        add_text: if text should be added to each cell with the given value
        labels: labels to add the x- and y-axis
    Returns:
        A tuple consisting of the figure and respective ax objects (or array of ax objects) of the generated
    """
    nb, n_classes, rows, cols = 1, confmat.shape[0], 1, 1

    if labels is not None and confmat.ndim != 3 and len(labels) != n_classes:
        raise ValueError(
            "Expected number of elements in arg `labels` to match number of labels in confmat but "
            f"got {len(labels)} and {n_classes}"
        )
    if confmat.ndim == 3:
        fig_label = labels or np.arange(nb)
        labels = list(map(str, range(n_classes)))
    else:
        fig_label = None
        labels = labels or np.arange(n_classes).tolist()

    fig, axs = plt.subplots(nrows=rows, ncols=cols) if ax is None else (ax.get_figure(), ax)
    axs = trim_axs(axs, nb)
    for i in range(nb):
        ax = axs[i] if rows != 1 and cols != 1 else axs
        if fig_label is not None:
            ax.set_title(f"Label {fig_label[i]}", fontsize=15)
        ax.imshow(confmat[i].cpu().detach() if confmat.ndim == 3 else confmat.cpu().detach())
        ax.set_xlabel("Predicted", fontsize=15)
        ax.set_ylabel("Actual", fontsize=15)
        ax.set_xticks(list(range(n_classes)))
        ax.set_yticks(list(range(n_classes)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_yticklabels(labels, rotation=0, fontsize=8)

        if add_text:
            for ii, jj in product(range(n_classes), range(n_classes)):
                val = confmat[i, ii, jj] if confmat.ndim == 3 else confmat[ii, jj]
                ax.text(jj, ii, str(val.item()), ha="center", va="center", fontsize=15)

    return fig, axs
