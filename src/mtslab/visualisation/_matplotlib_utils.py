import mplcursors


### Local
from mtslab.utils import *

###
###     Shorthand for matplotlib colors
###

_selection_from_CSS4_COLORS = [
    "navy",
    "teal",
    "mediumspringgreen",
    "coral",
    "firebrick",
    "peru",
    "olive",
    "red",
    "green",
    "palegreen",
    "steelblue",
    "lime",
    "yellow",
    "slateblue",
    "darkorange",
    "royalblue",
]

COLORS_DICT = {
    ### A few favourites
    **{c: mcolors.to_rgba(mcolors.CSS4_COLORS[c]) for c in _selection_from_CSS4_COLORS},
    ### colors from mcolors.TABLEAU_COLORS
    **{k: mcolors.to_rgba(c) for k, c in mcolors.TABLEAU_COLORS.items()},
}
COLORS = list(COLORS_DICT.values())

###
###     Matplotlib Helper: Miscellaneous helpful methods for matplotlib figures/axes
###


class MatplotlibHelper:
    """Matplotlib helper methods."""
    def _full_screen(self, fig: plt.Figure, tight_layout: bool = True) -> None:
        manager = fig.canvas.manager
        try:
            manager.window.showMaximized()
        except:
            fig.canvas.manager.full_screen_toggle()
        if tight_layout:
            def on_resize(event):
                fig.tight_layout(pad=0)
            fig.canvas.mpl_connect("resize_event", on_resize)

class MatplotlibHeatmapCursor:
    """Simple cursor for a seaborn heatmap."""
    def __init__(self, ax: plt.Axes, data_matrix: np.ndarray) -> None:
        self.data_matrix = data_matrix
        dummy = ax.imshow(data_matrix, zorder=-1, aspect="auto")
        cursor = mplcursors.cursor(dummy, hover=True)
        cursor.connect("add", self._show_annotation)

    def _show_annotation(self, sel):
        x = int(sel.target[0])
        y = int(sel.target[1])
        sel.annotation.set_text(f"{self.data_matrix[y, x]}")
        sel.annotation.get_bbox_patch().set(alpha=0.9)
