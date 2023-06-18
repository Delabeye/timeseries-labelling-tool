from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from matplotlib.collections import QuadMesh

### Local
from mtslab.utils._utils import *
from mtslab.visualisation._matplotlib_utils import *
from mtslab.representation import MultiHotEncoder

###
###     Matrix Plotter
###


class MatrixPlotter:
    """A plotter for matrices"""

    def __init__(
        self,
        matrix: np.ndarray | pd.DataFrame,
        continuous_axis: Sequence[int] = [],
    ) -> None:
        """Plot a matrix.

        Parameters
        ----------
        matrix : np.ndarray | pd.DataFrame
            Matrix to plot.
        continuous_axis : Sequence[int], optional
            Whether axis 0 (row), 1 (column) or both should be interpreted
             as continuous variables. This typically forces categorical variables
             to be shown by the tick Locator.
             by default []
        """
        self.matrix = pd.DataFrame(matrix)
        self.is_complex = np.iscomplex(matrix).any()
        self.continuous_axis = continuous_axis

    @staticmethod
    def _dummy_colobar(x, y, ax):
        """Create a dummy, non-visible colorbar for an Axes

        This hack is useful to make the Axes share
        the same width as the other Axes with colorbar.
        """
        from matplotlib.colors import LinearSegmentedColormap

        colors = [(1, 1, 1), (1, 1, 1)]  # white to white
        n_bin = 2
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bin)
        xcoord, _ = MatrixPlotter._get_coord(x)
        ycoord, _ = MatrixPlotter._get_coord(x)
        C = np.zeros((len(xcoord), len(ycoord)))
        pcm = ax.pcolormesh(xcoord, ycoord, C, cmap=cmap)
        fig = ax.get_figure()
        pcm_colorbar = fig.colorbar(pcm, ax=ax)
        pcm_colorbar.set_ticks([])
        pcm_colorbar.outline.set_visible(False)

    @staticmethod
    def _get_coord(x: np.ndarray, continuous: bool = True):
        """Get the coordinates for the pcolormesh"""
        flag = continuous and not np.issubdtype(x.dtype, np.number)
        if continuous and np.issubdtype(x.dtype, np.number):
            dx, x_start, x_end = np.diff(x).mean(), x[0], x[-1]
            return np.arange(x_start - dx / 2, x_end + dx, dx), flag
        else:
            return np.arange(-0.5, len(x), 1), flag

    @staticmethod
    def cmap_norm(
        matrix: pd.DataFrame | np.ndarray, norm: str | mcolors.Normalize = None
    ) -> dict[str, str | mcolors.Normalize]:
        """Set up the colormap and normalization for a matrix plot.

        Parameters
        ----------
        matrix : pd.DataFrame | np.ndarray
            matrix to apply cmap and norm on (as DataFrame or ndarray).
        norm : str | mcolors.Normalize, optional
            Normalization to apply to the colormap. It can be:
                - "zero_center": the colormap is centered at 0 and
                 goes from -max|matrix| to +max|matrix| or [min, 0] or [0, max]
                - "zero": the colormap is centered or lowered/raised to 0 and
                    goes from [min, max], [min, 0] or [0, max]
                - "log": the colormap is log-normalized
                - None: no normalization is applied (default, linear scale)
                - a mcolors.Normalize object: it is used to normalize the colormap.
             by default None

        Returns
        -------
        dict[str, str|mcolors.Normalize]
            Returns kwargs to pass on to pcolormesh or similar artist
            (e.g., {"cmap":"Reds", "norm":mcolors.LogNorm()})
        """

        ### Check norm is valid
        available_norms = [None, "zero", "zero_center", "log"]
        if not (isinstance(norm, mcolors.Normalize) or norm in available_norms):
            raise ValueError(
                f"Invalid norm '{norm}'. `norm` must be a"
                f" mcolors.Normalize object or one of {str(available_norms)}"
            )

        ### Compute norm as appropriate
        kw_pcm = {"cmap": "Reds"}  # set default
        matrix = np.array(matrix)
        zmin, zmax = matrix.min(), matrix.max()
        if norm in ["zero_center", "zero"]:
            # if data is one-sided (positive or negative), use a one-sided colormap
            if zmin * zmax >= 0:
                if zmin < 0 and zmax <= 0:
                    vmin, vmax = zmin, 0
                    kw_pcm["cmap"] = "Blues"
                elif zmin >= 0 and zmax > 0:
                    vmin, vmax = 0, zmax
                    kw_pcm["cmap"] = "Reds"
                kw_pcm["norm"] = mcolors.Normalize(vmin, vmax)
            # adapt the norm if data is two-sided
            else:
                kw_pcm["cmap"] = "RdBu_r"
                if norm == "zero_center":
                    kw_pcm["norm"] = mcolors.CenteredNorm(vcenter=0)
                elif norm == "zero":
                    kw_pcm["norm"] = mcolors.TwoSlopeNorm(
                        vmin=vmin, vcenter=0, vmax=vmax
                    )
        elif norm in ["log"]:
            kw_pcm["norm"] = mcolors.LogNorm()
            kw_pcm["cmap"] = "viridis"
        elif isinstance(norm, mcolors.Normalize):
            kw_pcm["norm"] = norm
        return kw_pcm

    def _ticks(
        self, matrix: pd.DataFrame, ax: plt.Axes, max_ticks: int = 15
    ) -> plt.Axes:
        """Add ticks to discrete axis (if any)). This function is for internal use only."""
        if 0 not in self.continuous_axis or self._xflag:
            xticks = list(map(str, matrix.columns))
            ax.set_xticks(range(len(xticks)), xticks)
            if len(xticks) > max_ticks:
                ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
            else:
                ax.xaxis.set_major_locator(plt.FixedLocator(range(len(xticks))))

        if 1 not in self.continuous_axis or self._yflag:
            yticks = list(map(str, matrix.index))
            ax.set_yticks(range(len(yticks)), yticks)
            if len(yticks) > max_ticks:
                ax.yaxis.set_major_locator(plt.MaxNLocator(max_ticks))
            else:
                ax.yaxis.set_major_locator(plt.FixedLocator(range(len(yticks))))

        return ax

    def _plot_raw(
        self,
        matrix: pd.DataFrame,
        ax: plt.Axes,
        projection: str = None,
        norm: str | mcolors.Normalize = None,
        cbar: bool = True,
        **kw_pcolormesh,
    ) -> QuadMesh:
        """Plot the raw matrix as pcolormesh. This function is for internal use only."""
        ### Set up the x and y coordinates
        # (raise a flag if the index is not both continuous and numeric)
        # this flag is then used to set the ticks.
        xcoord, self._xflag = self._get_coord(
            matrix.columns, continuous=0 in self.continuous_axis
        )
        ycoord, self._yflag = self._get_coord(
            matrix.index, continuous=1 in self.continuous_axis
        )

        ### Set up the colormap and normalization
        norm = norm or kw_pcolormesh.get("norm", None)
        kw_pcm = MatrixPlotter.cmap_norm(self.matrix, norm) | kw_pcolormesh

        if projection in ["3d", "3D"]:
            ### Generate the surface
            xcoord, ycoord = matrix.columns, matrix.index
            mesh = np.meshgrid(xcoord, ycoord)
            pcm = ax.plot_surface(*mesh, matrix, **kw_pcm)
        else:
            ### Generate the pcolormesh
            # pcm = ax.pcolormesh(ycoord[None, :], xcoord[:, None], matrix, **kw_pcm)
            pcm = ax.pcolormesh(xcoord[None, :], ycoord[:, None], matrix, **kw_pcm)

        ### Set up the colorbar
        if cbar:
            pcm_colorbar = plt.colorbar(pcm, ax=ax)
        return pcm

    ###
    ###     Plot as a collection of vectors
    ###

    def plot_as_vectors(
        self,
        vector_type: str = None,
        ax: plt.Axes = None,
        cbar: bool = True,
        norm: str | mcolors.Normalize = None,
        cmap: str = None,
        kw_pcolormesh: dict = {},
    ) -> QuadMesh:
        """Plot a matrix as a collection of vectors.

        Parameters
        ----------
        vector_type : str, optional
            Type of vectors among ["row", "column", None]. If None, no separator is shown
            by default None
        ax : plt.Axes, optional
            Axes to plot the matrix on, by default None
        cbar : bool, optional
            Whether to plot the colorbar, by default True
        norm : str | mcolors.Normalize, optional
            Normalization to apply to the colormap. It can be:
                - "zero_center": the colormap is centered at 0 and
                 goes from -max|matrix| to +max|matrix| or [min, 0] or [0, max]
                - "zero": the colormap is centered or lowered/raised to 0 and
                    goes from [min, max], [min, 0] or [0, max]
                - "log": the colormap is log-normalized
                - None: no normalization is applied (default, linear scale)
                - a mcolors.Normalize object: it is used to normalize the colormap.
             by default None
        cmap : str, optional
            Colormap (overwrites norm-induced cmap), by default None
        kw_pcolormesh : dict, optional
            Kwargs to pass on to ``pcolormesh``, by default {}

        Returns
        -------
        matplotlib.collections.QuadMesh
            Returns the pcolormesh
        """

        separator_format = dict(color="black", lw=0.5)
        ax = ax or plt.subplots()[1]

        cmap_norm = MatrixPlotter.cmap_norm(self.matrix, norm)
        kw_pcm = cmap_norm | {"cbar": cbar, "cmap": cmap} | kw_pcolormesh

        ### Plot raw
        self._plot_raw(self.matrix, ax=ax, **kw_pcm)
        self._ticks(self.matrix, ax=ax)

        ### Add vector separators
        if vector_type == "column":
            for sep_index in range(len(self.matrix.columns) + 1):
                ax.axvline(sep_index - 0.5, **separator_format)
        elif vector_type == "row":
            for sep_index in range(len(self.matrix.index) + 1):
                ax.axhline(sep_index - 0.5, **separator_format)
        return ax

    ###
    ###     Plot complex matrix
    ###

    def plot_complex(
        self,
        mode: str | Callable = "db",
        ax: plt.Axes = None,
        projection: str = None,
        cbar: bool = True,
        norm: str | mcolors.Normalize = None,
        cmap: str = None,
        kw_pcolormesh: dict = {},
    ) -> QuadMesh:
        """Plot a complex matrix with different modes.

        Parameters
        ----------
        mode : str | Callable, optional
            Modes to plot complex matrices, for each matrix in ["A", "X", "Y"], to choose from:
                - "ReIm": plot real and imaginary parts separately side by side
                - "MagPhase": plot magnitude and phase separately side by side
                - "Re": plot only the real part
                - "Im": plot only the imaginary part
                - "db": plot the power spectral density
                - "magnitude" or "abs": plot the magnitude
                - "angle": plot the angle
                - "phase": plot the phase (unwrapped angle)
                - None: no mode is applied (default).
            Default modes are: {"A": "ReIm", "X": "ReIm", "Y": "ReIm"}.
            by default {}
        ax : plt.Axes, optional
            Axes to plot the matrix on, by default None
        projection : str, optional
            Whether the plot should be in "3d", (projection type of `ax`, cf ``matplotlib.pyplot.axes``),
            by default None
        cbar : bool, optional
            Whether to plot the colorbar, by default True
        norm : str | mcolors.Normalize, optional
            Normalization to apply to the colormap. It can be:
                - "zero_center": the colormap is centered at 0 and
                 goes from -max|matrix| to +max|matrix| or [min, 0] or [0, max]
                - "zero": the colormap is centered or lowered/raised to 0 and
                    goes from [min, max], [min, 0] or [0, max]
                - "log": the colormap is log-normalized
                - None: no normalization is applied (default, linear scale)
                - a mcolors.Normalize object: it is used to normalize the colormap.
             by default None
        cmap : str, optional
            Colormap (overwrites norm-induced cmap), by default None
        kw_pcolormesh : dict, optional
            Kwargs to pass on to ``pcolormesh``, by default {}

        Returns
        -------
        matplotlib.collections.QuadMesh
            Returns the pcolormesh
        """

        cmap_norm = MatrixPlotter.cmap_norm(self.matrix, norm)
        kw_pcm = cmap_norm | {"cbar": cbar, "cmap": cmap} | kw_pcolormesh

        ### Split Re/Im parts
        if mode in ["ReIm", "MagPhase"]:
            if projection in ["3d", "3D"]:
                ax = (
                    ax
                    or plt.subplots(
                        1, 2, sharex=True, sharey=True, subplot_kw={"projection": "3d"}
                    )[1]
                )
            else:
                ax = ax or plt.subplots(1, 2, sharex=True, sharey=True)[1]
            ax[-1].get_figure().subplots_adjust(wspace=0)

            if mode in ["ReIm"]:
                df_left = self.matrix.apply(np.real)
                df_right = self.matrix.apply(np.imag)
            elif mode in ["MagPhase"]:
                df_left = self.matrix.apply(np.abs)
                df_right = self.matrix.apply(lambda x: np.unwrap(np.angle(x), axis=-1))

            if cbar:
                cbar_wrt_matrix = pd.concat([df_left, df_right], axis=1).values
                norm = MatrixPlotter.cmap_norm(cbar_wrt_matrix, norm)
                kw_pcm = cmap_norm | {"cmap": cmap} | kw_pcolormesh

            pcm = [
                self._plot_raw(
                    df_left,
                    ax=ax[0],
                    projection=projection,
                    **{**kw_pcm, **{"cbar": False}},
                ),
                self._plot_raw(
                    df_right,
                    ax=ax[1],
                    projection=projection,
                    **{**kw_pcm, **{"cbar": True}},
                ),
            ]
            self._ticks(df_left, ax=ax[0])
            self._ticks(df_right, ax=ax[1])

        ### Apply scalar mode on data
        else:
            if projection in ["3d", "3D"]:
                ax = ax or plt.subplots(subplot_kw={"projection": "3d"})[1]
            else:
                ax = ax or plt.subplots()[1]
            if mode is None:
                df = self.matrix
            elif isinstance(mode, Callable):
                df = self.matrix.apply(mode)
            elif mode in ["Re"]:
                df = self.matrix.apply(np.real)
            elif mode in ["Im"]:
                df = self.matrix.apply(np.imag)
            elif mode in ["db", "default"]:
                df = self.matrix.apply(lambda x: 20 * np.log10(np.abs(x)))
            elif mode in ["magnitude", "abs"]:
                df = self.matrix.apply(np.abs)
            elif mode in ["angle"]:
                df = self.matrix.apply(np.angle)
            elif mode in ["phase"]:
                df = self.matrix.apply(lambda x: np.unwrap(np.angle(x), axis=-1))
            else:
                raise ValueError(f"Unknown mode: {mode}")
            pcm = self._plot_raw(df, ax=ax, projection=projection, **kw_pcm)
        return pcm

    ###
    ###     Plot matrix as a time series, with optional clustering labels
    ###

    def plot_series(
        self,
        labels: np.ndarray | pd.DataFrame = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot a multivariate time series, optionally tagged with clustering labels.

        Parameters
        ----------
        labels : np.ndarray | pd.DataFrame, optional
            Labels either as post-clustering labels, i.e., 1d array (1, n_samples) or
            one-hot encoded ndarray (n_clusters, n_samples); or as post-subclustering labels,
            i.e., multi-hot encoded ndarray (n_subclusters, n_samples).
        ax : plt.Axes, optional
            If none passed, creates a new figure and Axes, by default None

        Returns
        -------
        plt.Axes
            Axes on which the plot was drawn.
        """
        series = pd.DataFrame(self.matrix)
        if labels is not None:
            MatrixPlotter._plot_with_labels(series, labels, ax=ax, **kwargs)
        else:
            MatrixPlotter._plot_without_labels(series, ax=ax, **kwargs)
        return ax

    @staticmethod
    def _plot_without_labels(
        series: pd.DataFrame, ax: plt.Axes = None, **kwargs
    ) -> plt.Axes:
        ax = ax or plt.subplots()[1]

        ### Plot the series
        alphas = np.linspace(1.0, 0.5, len(series.columns))
        ax_twins = [ax] + [ax.twinx() for _ in range(len(series.columns) - 1)]
        ax_locs = (
            [
                "upper left",
                "upper right",
                "lower left",
                "lower right",
                "upper center",
                "lower center",
            ]
            * len(series.columns)
        )[: len(ax_twins)]
        for i_col, (col, alpha, twin, loc) in enumerate(
            zip(series.columns, alphas, ax_twins, ax_locs)
        ):
            twin.plot(
                series.index,
                series[col],
                alpha=alpha,
                label=col,
                color=COLORS[i_col % len(COLORS)],
                **kwargs,
            )
            vmax = np.abs(series[col]).max()
            twin.set_ylim(-vmax * 1.05, vmax * 1.05)
            twin.set_ylabel(col)
            twin.legend(loc=loc)

        ax.plot()
        ax.set_xlabel(series.index.name)
        ax.legend(loc=ax_locs[0])
        ### Set transparent background for cursors & such
        ax.set_zorder(np.inf)
        ax.set_facecolor("none")
        return ax

    @staticmethod
    def _plot_with_labels(
        series: pd.DataFrame, labels: np.ndarray, ax: plt.Axes = None
    ) -> plt.Axes:
        ax = ax or plt.subplots()[1]

        ### One-Hot encode labels, and interpolate to match the series index
        labels = np.array(labels)
        if labels.ndim == 1:
            # labels = encode_onehot(labels)
            labels = MultiHotEncoder(labels).to_onehot()
        elif labels.ndim == 2:
            if not set(np.unique(labels)).issubset({0, 1}):
                raise ValueError(
                    "Clustering display on series is only available for binary labels "
                    "at the moment. That is, label repetition in a combination is not."
                )  # TODO enable label multiplicity (requires base `base` multi-hot encoding)
            labels = MultiHotEncoder(labels).to_onehot()

        labels_index = np.linspace(series.index[0], series.index[-1], labels.shape[1])
        labels = scipy.interpolate.interp1d(labels_index, labels)(series.index)
        labels = np.rint(labels).astype(int)

        ### Plot the series
        alphas = np.linspace(1.0, 0.5, len(series.columns))
        ax_twins = [ax] + [ax.twinx() for _ in range(len(series.columns) - 1)]
        for col, alpha, twin in zip(series.columns, alphas, ax_twins):
            # apply color where cluster is active
            # (on twin, for multiple axes)
            # NOTE color is overwritten if multiple clusters are active simultaneously
            for cluster_id, label in enumerate(labels):
                cluster_name = f"[{chr(cluster_id + 65)}]"
                mask = np.isclose(label, 0)
                twin.plot(
                    np.ma.masked_where(mask, series.index),
                    np.ma.masked_where(mask, series[col]),
                    alpha=alpha,
                    label=cluster_name,
                    color=COLORS[cluster_id % len(COLORS)],
                )
            vmax = np.abs(series[col]).max()
            twin.set_ylim(-vmax * 1.05, vmax * 1.05)
            twin.set_ylabel(col)
        ax.plot()
        ax.set_xlabel(series.index.name)
        ax.legend(ncol=len(series.columns), loc="upper right")
        return ax


class LinearMatrixFactorisationPlotter:
    """Plotter for Linear Matrix Factorisation"""

    def plot_A(self, kw_pcolormesh: dict = {}, **kwargs) -> plt.Axes:
        kw_pcm = {"cmap": "viridis"} | kw_pcolormesh
        kw_plot = {"vector_type": "column", "kw_pcolormesh": kw_pcm} | kwargs
        return MatrixPlotter(self.A_, continuous_axis=[1]).plot_as_vectors(**kw_plot)

    def plot_X(self, kw_pcolormesh: dict = {}, **kwargs) -> plt.Axes:
        kw_pcm = {"cmap": "Reds"} | kw_pcolormesh
        kw_plot = {"vector_type": "row", "kw_pcolormesh": kw_pcm} | kwargs
        return MatrixPlotter(self.X_.T, continuous_axis=[0]).plot_as_vectors(**kw_plot)

    def plot_Y(self, kw_pcolormesh: dict = {}, **kwargs) -> plt.Axes:
        kw_pcm = {"cmap": "viridis"} | kw_pcolormesh
        kw_plot = {"kw_pcolormesh": kw_pcm} | kwargs
        return MatrixPlotter(self.Y_, continuous_axis=[0, 1]).plot_as_vectors(**kw_plot)

    def plot(
        self, kw_A: dict = {}, kw_X: dict = {}, kw_Y: dict = {}
    ) -> dict[str, plt.Axes]:
        """Plot the linear matrix factorisation"""
        layout1 = """
                ayy
                ayy
                .xx
        """
        fig, ax = plt.subplot_mosaic(layout1)
        self.plot_A(kw_A, ax=ax["a"])
        self.plot_X(kw_X, ax=ax["x"])
        self.plot_Y(kw_Y, ax=ax["y"])
        if report := True:
            fig.text(0.01, 0.00, self.__repr__(), ha="left", va="bottom", wrap=True)
        fig.tight_layout()
        self.fig, self.ax = fig, ax
        return ax
