"""

    Data labelling utilities
    (Labelliser - from dataframe to multi-labels)
    
    NOTE overcome limits in TRAINSET and Label-Studio 
    in terms of dataset size and multi-label capabilities.

"""
from __future__ import annotations

from matplotlib.backend_bases import Event
from matplotlib.widgets import Cursor, Button, TextBox
from tkinter import filedialog as fd

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

### Local
from mtslab.utils import *
from mtslab.visualisation import COLORS, MatrixPlotter

_PAUSE = 1e-2  # drawing pause time for plotting routines (in seconds)
_EVENT_PAUSE = 1e-4  # event pause time (in seconds)


class Labeller:
    """Matplotlib-based time series labelling tool.

    Attributes
    ----------
    data : pd.DataFrame | Sequence[pd.DataFrame]
        Data to label.
    save_to : str | Path
        Path to save labels to.
    label_names : Sequence[str]
        Names of labels.

    labels_ : pd.DataFrame
        Labels (as a boolean dataframe).
    labels_compressed_ : dict
        Compressed labels (as a dict of time ranges).

    Methods
    -------
    save
        Save labels to file (with backup).
    load
        Load labels from file.
    run
        Run labelling tool (`plt.show()`).
    """

    def __init__(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
        save_to: str | Path,
        label_names: Sequence[str],
        kw_specgram: dict = {},
    ) -> None:
        self.data = [data] if isinstance(data, pd.DataFrame) else list(data)
        self.save_to = save_to
        self.label_names = label_names
        self._kw_specgram = kw_specgram

        self._df = self.data[0]  # main dataframe (for index, etc.)

        ### Setup figure, color, font, icons
        self.fig = plt.figure(str(self.save_to))
        self.ax = self.fig.subplots(2, sharex=True)
        (self._i_ax_labels, (self.ax_labels)), (
            self._i_ax_data,
            (self.ax_data),
        ) = enumerate(self.ax)

        self._data_artist = None
        self._default_font_size = 12
        path_to_icons = Path(__file__).parent / "icons"
        self._icons = {
            icon_name: plt.imread(path_to_icons / f"{icon_name}.png")
            for icon_name in [
                "locked",
                "unlocked",
                "update",
                "spectrogram",
                "timeseries",
            ]
        }

        ### Setup data plotter
        self.setup_plotter()

        ### Setup label plotter
        labels = pd.DataFrame(
            {name: 0 * self._df.index for name in label_names}, index=self._df.index
        ).astype(bool)
        self.label_plotter = LabelPlotter(labels=labels, ax=self.ax_labels)

        ### Setup SWITCH VIEW (data)
        # switch data button
        self._switchdatabtn_ax = plt.axes([0.95, 0.16, 0.045, 0.07])
        self._switchdatabtn = Button(
            self._switchdatabtn_ax, "", image=self._icons["timeseries"]
        )
        self._switchdatabtn.on_clicked(self._switch_view_data)
        self._switchdatabtn.label.set_fontsize(self._default_font_size)

        ### Setup SWITCH VIEW (spectrogram)
        # switch data button
        self._switchspecgrambtn_ax = plt.axes([0.95, 0.1, 0.045, 0.05])
        self._switchspecgrambtn = Button(
            self._switchspecgrambtn_ax, "", image=self._icons["spectrogram"]
        )
        self._switchspecgrambtn.on_clicked(self._switch_view_specgram)
        self._switchspecgrambtn.label.set_fontsize(self._default_font_size)

        ### Setup TICK/UNTICK (segmentation)
        self._tick_mode = "untick" # default is "tick"
        self._xticks = [self._df.index[0], self._df.index[-1]]
        self._xticks_vlines = [None, None]
        # tick & untick events
        self._tick_action = self.fig.canvas.mpl_connect(
            "button_press_event", self._tick_onclick
        )
        self._untick_action = self.fig.canvas.mpl_connect(
            "button_press_event", self._untick_onclick
        )
        # tick & untick & lock (ticking) button
        self._tickbtn_ax = plt.axes([0.95, 0.28, 0.045, 0.1])
        self._tickbtn = Button(self._tickbtn_ax, "")
        self._toggle_tick_mode()
        self._tickbtn.on_clicked(self._toggle_tick_mode)
        # LOCK (ticking) button
        self._tick_locked = True # default is unlocked (False)
        self._lock_tickbtn_ax = plt.axes([0.95, 0.4, 0.045, 0.05])
        self._lock_tickbtn = Button(self._lock_tickbtn_ax, "")
        self._toggle_lock_tick_mode()
        self._lock_tickbtn.on_clicked(self._toggle_lock_tick_mode)


        ### Setup LABEL (labelling)
        self._label_mode = None
        # label event
        self._label_action = self.fig.canvas.mpl_connect(
            "button_press_event", self._label_onclick
        )
        # label & lock button
        self._labelbtn_ax = plt.axes([0.9, 0.71, 0.095, 0.1])
        self._labelbtn = Button(self._labelbtn_ax, "")
        self._toggle_label_mode()
        self._labelbtn.on_clicked(self._toggle_label_mode)

        ### Setup button SAVE
        self.autosave = None  # seconds (disabled if None)
        self._time_last_save = time.time()
        # save button
        self._savebtn_ax = plt.axes([0.9, 0.88, 0.095, 0.05])
        self._savebtn = Button(
            self._savebtn_ax,
            "save",
            color=mcolors.to_rgba("tab:orange", 0.8),
            hovercolor=mcolors.to_rgba("tab:orange", 0.4),
        )
        self._savebtn.on_clicked(self._save_labels_event)
        self._savebtn.label.set_fontsize(self._default_font_size)

        ### Setup button LOAD
        # load button
        self._loadbtn_ax = plt.axes([0.9, 0.94, 0.095, 0.05])
        self._loadbtn = Button(
            self._loadbtn_ax, "load labels", color=mcolors.to_rgba("tab:gray", 0.8)
        )
        self._loadbtn.on_clicked(self._load_labels_event)
        self._loadbtn.label.set_fontsize(self._default_font_size)

    @property
    def labels_(self):
        if hasattr(self, "label_plotter"):
            return self.label_plotter.df

    @property
    def labels_compressed_(self):
        return compress_binary_matrix(self.labels_)

    ###
    ###     Events
    ###

    def _event(method):
        def wrapper(self, event: Event = None):
            method(self, event)
            self._on_any_event(event)

        return wrapper

    def _on_any_event(self, event: Event = None):
        """Actions to perform each time an event occurs."""
        ### Autosave
        if (
            getattr(self, "autosave", False)
            and (t := time.time()) - self._time_last_save > 60
        ):
            self.save()
            self._time_last_save = t
        ### Toggle save button if labels are not up-to-date
        self._toggle_save_uptodate_btn()

    @_event
    def _switch_view_data(self, event: Event = None):
        self._i_data += 1  # increment dataframe index
        self._i_column = 0  # reset column index
        self.ax_data.clear()
        self._view_data()
        if hasattr(self, "_xticks"):
            self._redraw_xticks()
        plt.pause(_PAUSE)

    @_event
    def _switch_view_specgram(self, event: Event = None):
        self._i_column += 1  # increment column index
        self.ax_data.clear()
        ic([ax for ax in self.fig.get_axes() if hasattr(ax, "ylabel")])
        self._view_specgram()
        if hasattr(self, "_xticks"):
            self._redraw_xticks()
        plt.pause(_PAUSE)

    def _load_labels_event(self, event: Event = None):
        self.load(None)

    @_event
    def _save_labels_event(self, event: Event = None):
        """Save labels to file (with backup)."""
        if not self.label_plotter.save_uptodate:
            self.save(self.save_to, backup=False)

    @_event
    def _toggle_label_mode(self, event: Event = None):
        """Toggle between labelling modes (circular shift)."""
        modes = ["label", None]
        self._label_mode = modes[(modes.index(self._label_mode) + 1) % len(modes)]
        if self._label_mode == "label":
            self._labelbtn.color = mcolors.to_rgba("xkcd:turquoise")
            self._labelbtn.hovercolor = mcolors.to_rgba("xkcd:turquoise", alpha=0.5)
            self._labelbtn.label.set_text("Label")
        elif self._label_mode is None:
            self._labelbtn.color = mcolors.to_rgba("tab:gray", 0.8)
            self._labelbtn.hovercolor = mcolors.to_rgba("tab:gray", 0.4)
            self._labelbtn.label.set_text("Labelling\nlocked")
        self._labelbtn.label.set_fontsize(self._default_font_size)

    @_event
    def _label_onclick(self, event: Event = None):
        """Label closest label upon clicking."""
        if self._label_mode == "label":
            if event.inaxes == self.ax_labels:
                ix, iy = event.xdata, event.ydata
                self.label_plotter.update((ix, iy), self._xticks)

    def _force_update_labels(self, event: Event = None):
        """Force update of label plotter."""
        self._reset_between_xticks()

    @_event
    def _tick_onclick(self, event: Event = None):
        """Add new tick and vline upon clicking."""
        if self._tick_mode == "tick" and not self._tick_locked:
            if event.inaxes in [self.ax_data]:
                self._add_xticks([event.xdata])

    @_event
    def _untick_onclick(self, event: Event = None):
        """Delete closest tick and vlines upon clicking."""
        if self._tick_mode == "untick" and len(self._xticks) > 2 and not self._tick_locked:
            if event.inaxes in [self.ax_data]:
                ix = event.xdata
                idx = np.nanargmin(np.abs(np.array(self._xticks[1:-1]) - ix)) + 1
                del self._xticks[idx]
                for vline in self._xticks_vlines[idx]:
                    vline.remove()
                del self._xticks_vlines[idx]
            self._force_update_labels()

    @_event
    def _toggle_lock_tick_mode(self, event: Event = None):
        """Toggle between tick locking modes (circular shift)."""
        self._tick_locked = not self._tick_locked
        if self._tick_locked is True:
            self._lock_tickbtn.color = mcolors.to_rgba("xkcd:red")
            self._lock_tickbtn.hovercolor = mcolors.to_rgba("xkcd:red", 0.5)
            self._lock_tickbtn.label.set_text("Lock")
            plt.pause(_EVENT_PAUSE)
        elif self._tick_locked is False:
            self._lock_tickbtn.color = mcolors.to_rgba("xkcd:green")
            self._lock_tickbtn.hovercolor = mcolors.to_rgba("xkcd:green", 0.5)
            self._lock_tickbtn.label.set_text("Unlocked")
            plt.pause(_EVENT_PAUSE)

    @_event
    def _toggle_tick_mode(self, event: Event = None):
        """Toggle between ticking modes (circular shift)."""
        modes = ["tick", "untick"]
        self._tick_mode = modes[(modes.index(self._tick_mode) + 1) % len(modes)]
        if self._tick_mode == "tick":
            self._tickbtn.color = mcolors.to_rgba("xkcd:turquoise")
            self._tickbtn.hovercolor = mcolors.to_rgba("xkcd:turquoise", 0.5)
            self._tickbtn.label.set_text("Tick")
            plt.pause(_EVENT_PAUSE)
        elif self._tick_mode == "untick":
            self._tickbtn.color = mcolors.to_rgba("xkcd:coral")
            self._tickbtn.hovercolor = mcolors.to_rgba("xkcd:coral", 0.5)
            self._tickbtn.label.set_text("Untick")
            plt.pause(_EVENT_PAUSE)

    def _toggle_save_uptodate_btn(self):
        if hasattr(self, "_savebtn"):
            if self.label_plotter.save_uptodate:
                if self._savebtn.color != mcolors.to_rgba("xkcd:green", alpha=0.8):
                    self._savebtn.color = mcolors.to_rgba("xkcd:green", alpha=0.8)
                    plt.pause(_EVENT_PAUSE)
            else:
                if self._savebtn.color != mcolors.to_rgba("xkcd:orange", alpha=0.8):
                    self._savebtn.color = mcolors.to_rgba("xkcd:orange", alpha=0.8)
                    plt.pause(_EVENT_PAUSE)

    ###
    ###     Data Plotter
    ###

    @property
    def current_data(self):
        self._i_data = self._i_data % len(self.data)
        return self.data[self._i_data]

    @property
    def current_column(self):
        self._i_column = self._i_column % len(self.current_data.columns)
        return self.current_data.columns[self._i_column]

    def _view_data(self):
        MatrixPlotter(self.current_data).plot_series(ax=self.ax_data)

    def _view_specgram(self):
        self.ax_data.specgram(
            self.current_data[self.current_column].values,
            **self._kw_specgram[self.current_column],
        )
        self.ax_data.legend([self.current_column], loc="upper right")

    def setup_plotter(self):
        """Setup figure"""
        self._i_data = -1  # current dataframe
        self._i_column = 0  # current column in dataframe (for column-focused features)

        ### Setup data plot
        matplotlib.rc("font", **{"size": 12})
        self._switch_view_data()
        self.ax[-1].set_xlim((self._df.index[0], self._df.index[-1]))

        ### Setup cursor (on data plot)
        self._cursor = Cursor(self.ax_data, useblit=True, color="black", linewidth=1)

        ### Full screen, adjust margins
        plt.get_current_fig_manager().window.showMaximized()
        plt.subplots_adjust(left=0.04, right=0.895, bottom=0.06, top=0.99, hspace=0.1)

    def _reset_between_xticks(self):
        """Reset labels between ticks to the value (True/False)
        with the most samples, column-wise.

        This is useful when the ticks that served to produce the labels
        have been erased in the meantime.
        """
        for col in self.label_plotter.df.columns:
            for x_pre, x_nxt in zip(self._xticks[:-1], self._xticks[1:]):
                sub_df_col = self.label_plotter.df[col][x_pre:x_nxt]
                reset_to = np.round(sub_df_col.sum() / sub_df_col.count()).astype(bool)
                self.label_plotter.df[col][x_pre:x_nxt] = reset_to
        self.label_plotter.update_barh()

    def _draw_vline(self, ax: plt.Axes | Sequence[plt.Axes], xtick: float, **kwargs):
        kw_axvline = {
            self.ax_data: dict(c="pink", alpha=0.7),
            self.ax_labels: dict(c="gray", alpha=0.3),
        }
        if isinstance(ax, plt.Axes):
            return ax.axvline(x=xtick, **kw_axvline[ax])
        else:
            return [axes.axvline(x=xtick, **kw_axvline[axes]) for axes in ax]

    def _redraw_xticks(self):
        """Redraw xticks/vlines."""
        for xtick, vlines in zip(self._xticks, self._xticks_vlines):
            if vlines is not None and len(vlines) == len(self.ax):
                vlines[self._i_ax_data] = self._draw_vline(self.ax_data, xtick)

    def _add_xticks(self, new_xticks: Sequence = []):
        """Add new xticks/vlines (coordinates) to the list of xticks if not already present"""
        if hasattr(self, "_xticks"):
            ### Add new xticks
            for xtick in new_xticks:
                if xtick not in self._xticks:
                    self._xticks.append(xtick)
                    self._xticks_vlines.append(self._draw_vline(self.ax, xtick))
            plt.pause(_PAUSE)
            self._sort_xticks()

    def _sort_xticks(self):
        """Sort xticks."""
        zip_xticks = list(zip(self._xticks, self._xticks_vlines))
        zip_xticks.sort()
        self._xticks, self._xticks_vlines = map(list, zip(*zip_xticks))

    ###
    ###     Handle labels
    ###

    def _fetch_data(self, filetype: str = "label"):
        """Fetch labels or data from file."""
        if filetype == "label":
            filetypes = [("pickle files", "*.pkl")]
        elif filetype == "data":
            filetypes = [("csv files", "*.csv")]
        filename = fd.askopenfilename(
            initialdir=Path(self.save_to).parent,
            title=f"Select {filetype} file",
            filetypes=filetypes,
        )
        if filename:
            if filetype == "label":
                return pickle.load(open(filename, "rb"))
            elif filetype == "data":
                return pd.read_csv(filename, index_col=0)

    def load(
        self,
        load_from: pd.DataFrame | dict[Any, tuple[int, int]] | Path | str = None,
        erase: bool = False,
    ):
        """Load labels from file or pre-loaded labels (as DataFrame or dict).

        Parameters
        ----------
        load_from : pd.DataFrame | dict[Any, tuple[int, int]] | Path | str, optional
            Load labels from file or pre-loaded labels (as DataFrame with the right index,
            or dict of time ranges (with time indices in the expected time index)).
            If None, load from file (with file dialog),
            by default None
        erase : bool, optional
            Erase current labels before loading new ones.
            If False, this will only replace common columns,
            by default False

        Raises
        ------
        ValueError
            When `load_from` is not a Path, str, DataFrame, dict or None.
        """

        ### Fetch labels
        if isinstance(load_from, Path | str) and str(load_from).endswith(".pkl"):
            labels_asdict = pickle.load(open(load_from, "rb"))
            labels = decompress_binary_matrix(
                labels_asdict, index=self._df.index, dtype=bool
            )
        elif isinstance(load_from, pd.DataFrame):
            labels = load_from.astype(bool)
            labels_asdict = compress_binary_matrix(labels)
        elif isinstance(load_from, dict):
            labels_asdict = load_from
            labels = decompress_binary_matrix(
                labels_asdict, index=self._df.index, dtype=bool
            )
        elif load_from is None:
            labels_asdict = self._fetch_data(filetype="label")
            if labels_asdict is not None:
                labels = decompress_binary_matrix(
                    labels_asdict, index=self._df.index, dtype=bool
                )
            else:
                return
        else:
            raise ValueError(
                "Expected Path, str, DataFrame or None,"
                f"got {type(load_from)} instead."
            )

        ### Erase current labels (erase previous labels, fill with the new labels)
        if erase:
            self.label_plotter.df.drop(
                columns=self.label_plotter.df.columns, axis=1, inplace=True
            )

        ### Integrate new labels into current label plotter
        for col in labels.columns:
            self.label_plotter.df[col] = labels[[col]]

            ### Update xticks
            new_xticks = [
                self._df.index[t]
                for rranges in labels_asdict.values()
                for rrange in rranges
                for t in rrange
                if self._df.index[t] not in self._xticks
            ]
            self._add_xticks(new_xticks)
        self.label_plotter.update_barh()

    def rename_labels(self, columns: Sequence[str] | dict):
        """Rename labels."""
        if isinstance(columns, dict):
            self.label_plotter.df.rename(columns=columns, inplace=True)
            self.label_names = list(columns.keys())
        elif isinstance(columns, Sequence):
            self.label_plotter.df.columns = columns
            self.label_names = columns
        self.label_plotter.update_barh()
        self.label_plotter.update_yticks()

    def save(self, save_to: Path | str = None, backup: bool = False):
        self.label_plotter.update_barh()  # reload barh artists
        self.label_plotter.save(save_to or self.save_to, backup=backup)
        if save_to == self.save_to:
            self.label_plotter.save_uptodate = True

    def run(self):
        plt.show()


class LabelPlotter:
    def __init__(self, labels: pd.DataFrame, ax: plt.Axes) -> None:
        self.df = labels.astype(bool)
        self.ax = ax
        self.save_uptodate = False

        self._bar_thickness = 0.5

        self._barh_artists = {}

        ### Set horizontal delimitations between labels
        for id in range(len(self.df.columns)):
            self.ax.axhline(y=id + self._bar_thickness, c="black")
            self.ax.axhline(y=id - self._bar_thickness, c="black")

        ### Setup ticks
        self.ax.set_ylim((-0.5, max(0.5, len(self.df.columns) - 0.5)))
        self.ax.set_yticks(np.arange(len(self.df.columns)))
        self.ax.set_yticklabels(self.df.columns)

    def ymin(self, id: int):
        return id - self._bar_thickness

    def ymax(self, id: int):
        return id + self._bar_thickness

    def update(self, loc: tuple[float, float], xticks: Sequence[float]):
        """Update label plotter."""

        self.save_uptodate = False  # reset uptodate flag

        ### Get the nearest delimiters
        x, y = loc
        xstart, xstop = get_neighbours_in_list(x, xticks)
        id, col = [
            (id, col)
            for id, col in enumerate(self.df.columns)
            if self.ymin(id) < y < self.ymax(id)
        ].pop()

        ### Update label values in the dataframe: invert values
        # exclude xstop, except when the last row is affected
        if xstop >= self.df.index[-1]:
            xstop += 1e-10  # add a small offset to include the last row
        self.df.loc[xstart:xstop, col] = np.invert(self.df.loc[xstart:xstop, col])
        self.update_barh(columns=[col])

    def update_barh(self, columns: Sequence[str] = None):
        """Update barh artists. If columns is None, update all columns."""
        if not columns:
            columns = self.df.columns
        for col in columns:
            id = self.df.columns.get_loc(col)

            ### Slice current column into active segments
            slices = compress_binary_matrix(
                self.df[col],
                include_window_size=True,
                squeeze=True,
                use_index_scale=True,
            )

            ### Clean
            if id in self._barh_artists:
                self._barh_artists[id].remove()
                del self._barh_artists[id]

            if slices:
                ### Draw coloured bars (active labels)
                self._barh_artists[id] = self.ax.broken_barh(
                    list(map(lambda sslice: (sslice[0], sslice[2]), slices)),
                    (id - self._bar_thickness, 2 * self._bar_thickness),
                    facecolors=COLORS[id],
                )

            plt.pause(_PAUSE)

    def update_yticks(self):
        """Update yticks."""
        self.ax.set_yticks(np.arange(len(self.df.columns)))
        self.ax.set_yticklabels(self.df.columns)
        plt.pause(_PAUSE)

    def save(self, filename: str | Path, backup: bool = False):
        labels_asdict = compress_binary_matrix(self.df)
        pickle.dump(labels_asdict, open(filename, "wb"))
        if backup:
            shutil.copyfile(filename, str(filename) + ".backup")
        self.save_uptodate = True
