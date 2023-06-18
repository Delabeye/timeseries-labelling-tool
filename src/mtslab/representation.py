"""

    Multi-label representations (multi-hot, one-hot, 1d).

"""

from sklearn.preprocessing import OneHotEncoder

### Local
from mtslab.utils import *

class MultiHotEncoder:
    """Multi-hot encoder.

    Attributes
    ----------
    data : np.ndarray | pd.DataFrame
        Multi-hot encoded input data as np.ndarray (n_components, n_samples)
        or pd.DataFrame (n_samples, n_components).
    multilabel_names : Sequence[str]
        Names of the multi-hot encoded labels (single components).
    label_names : Sequence[str]
        Names of the labels (combinations of single components).
    id2comb : dict[int, tuple[int]]
        Mapping from label ids to label combinations.
    comb2id : dict[tuple[int], int]
        Mapping from label combinations to label ids.
    id2name : dict[int, str]
        Mapping from label ids to label names (combinations of single components).
    name2id : dict[str, int]
        Mapping from label names (combinations of single components) to label ids.
    comb2onehot : dict[tuple[int], tuple[int]]
        Mapping from label combinations to one-hot encoded labels.
    onehot2comb : dict[tuple[int], tuple[int]]
        Mapping from one-hot encoded labels to label combinations.
    """

    def __init__(self, data: np.ndarray | pd.DataFrame = None) -> None:
        """Multi-hot encoder.

        Parameters
        ----------
        data : np.ndarray | pd.DataFrame
            Multi-hot encoded input data as np.ndarray (n_components, n_samples)
            or pd.DataFrame (n_samples, n_components).

        Raises
        ------
        TypeError
            When `data` is neither a pd.DataFrame or a np.ndarray.
        """

        self.data = data

        if self.data is not None:
            self._setup()

    @property
    def array(self):
        """Multi-hot encoded data as np.ndarray (n_components, n_samples)."""
        return self._df.values.T

    @property
    def df(self):
        """Multi-hot encoded data as pd.DataFrame (n_samples, n_components)."""
        return self._df

    @property
    def label_names(self):
        """Label names (combinations of single components)."""
        return list(self.id2name.values())

    @property
    def multilabel_names(self):
        """Names of the multi-hot encoded labels (single components)."""
        return self._df.columns

    def __repr__(self) -> str:
        return f"MultiHotEncoder(\n{self.df}\n)"

    def _setup(self):
        ### Cast to dataframe, save original type
        if isinstance(self.data, pd.DataFrame):
            self._df = self.data.astype(int)
            self.default_type = "DataFrame"
        elif isinstance(self.data, np.ndarray):
            self._df = pd.DataFrame(self.data.T).astype(int)
            self.default_type = "ndarray"
        else:
            raise TypeError(f"Unknown type: {type(self.data)}")

        ### Set meta data
        self._set_meta()

    def _set_meta(self):
        combinations = np.unique(self._df.values, axis=0)
        names = [
            MultiHotEncoder.encode_multilabel_name(self._df.columns, multiplicities)
            for multiplicities in combinations
        ]
        self.id2comb = {id: comb for id, comb in enumerate(combinations)}
        self.comb2id = {tuple(comb): id for id, comb in self.id2comb.items()}
        self.id2name = {id: col for id, col in enumerate(names)}
        self.name2id = {name: id for id, name in self.id2name.items()}
        self.comb2onehot = {
            tuple(comb): tuple(np.eye(1, combinations.shape[0], i).squeeze(0))
            for i, comb in enumerate(combinations)
        }
        self.onehot2comb = {
            tuple(onehot): tuple(comb) for comb, onehot in self.comb2onehot.items()
        }

    def _return_as(
        self, df: pd.DataFrame, astype: str = None, _squeeze: bool = False
    ) -> np.ndarray | pd.DataFrame:
        if astype == "DataFrame" or (
            astype is None and self.default_type == "DataFrame"
        ):
            return df
        elif astype == "ndarray" or (astype is None and self.default_type == "ndarray"):
            if _squeeze:
                return df.values.T.squeeze()
            else:
                return df.values.T

    def to_onehot(self, astype: str = None) -> np.ndarray | pd.DataFrame:
        """Expand multi-hot encoded matrix to one-hot encoded matrix.

        Parameters
        ----------
        astype : str, optional
            Output type (values in ["DataFrame", "ndarray"]).
            If None, return as `data`'s type as passed on to the class originally,
            by default None

        Returns
        -------
        np.ndarray | pd.DataFrame
            Multi-hot encoded data.
        """
        # Compute output columns (format column name with multiplicity)
        out_columns = list(self.id2name.values())
        # Compute output values (convert multihot labels to one-hot)
        out_values = np.array([self.comb2onehot[tuple(row)] for row in self._df.values])
        out = pd.DataFrame(out_values, index=self._df.index, columns=out_columns)
        return self._return_as(out, astype=astype)

    def to_1d(self, astype: str = None) -> np.ndarray | pd.DataFrame:
        X_as_onehot = self.to_onehot(astype="DataFrame")
        X_1d = pd.DataFrame(
            np.full(len(X_as_onehot), np.nan),
            index=X_as_onehot.index,
            columns=["_label_"],
        )
        ### Build 1d labels, attach the names to the dataframe
        for id, col in enumerate(X_as_onehot.columns):
            X_1d.iloc[X_as_onehot[col] == 1, 0] = id
        return self._return_as(X_1d, astype=astype, _squeeze=True)

    def from_1d(
        self, labels1d: np.ndarray | pd.DataFrame, columns: Sequence = None
    ) -> Self:
        """One-hot encode 1d labels (n_samples,).
        One-hot encoded labels are a special type of multi-hot encoded labels.

        Parameters
        ----------
        labels1d : np.ndarray | pd.DataFrame
            1d labels to encode (convert and load as one-hot).
        columns : Sequence, optional
            Names of the columns, by default None

        Returns
        -------
        Self
            Returns the instance itself.
        """

        index = labels1d.index if isinstance(labels1d, pd.DataFrame) else None
        values = (
            labels1d.values
            if isinstance(labels1d, pd.DataFrame)
            else np.array(labels1d).reshape(-1, 1)
        )
        data_asarray = OneHotEncoder(sparse_output=False).fit_transform(values)
        data = pd.DataFrame(data_asarray, index=index, columns=columns)
        self.data = data if isinstance(labels1d, pd.DataFrame) else data.values.T
        self._setup()
        return self

    def from_onehot(self, labels_onehot: np.ndarray | pd.DataFrame) -> Self:
        """Load one-hot encoded labels.
        One-hot encoded labels are a special type of multi-hot encoded labels.

        Parameters
        ----------
        labels_onehot : np.ndarray | pd.DataFrame
            One-hot encoded labels (as array (n_samples, n_components) or as DataFrame)
            to encode (simply load as one-hot).

        Returns
        -------
        Self
            Returns the instance itself.
        """
        self.data = (
            labels_onehot
            if isinstance(labels_onehot, pd.DataFrame)
            else np.array(labels_onehot)
        )
        self._setup()
        return self

    def from_multihot(self, labels_multihot: np.ndarray | pd.DataFrame) -> Self:
        """Load multi-hot encoded labels.

        Parameters
        ----------
        labels_multihot : np.ndarray | pd.DataFrame
            Multi-hot encoded labels (as array (n_components, n_samples) or as DataFrame)
            to encode (simply load as multi-hot).

        Returns
        -------
        Self
            Returns the instance itself.
        """
        self.data = (
            labels_multihot
            if isinstance(labels_multihot, pd.DataFrame)
            else np.array(labels_multihot)
        )
        self._setup()
        return self

    def inverse_transform(
        self, data: np.ndarray | pd.DataFrame, astype: str = None
    ) -> np.ndarray | pd.DataFrame:
        """Invert one-hot encoded or 1d labels to multi-hot encoded labels.

        Parameters
        ----------
        data : np.ndarray | pd.DataFrame
            One-hot encoded or 1d labels to invert.
            If 1d labels are provided as a DataFrame, there should be a single
            column named "_label_".
        astype : str, optional
            Output type (values in ["DataFrame", "ndarray"]).
            If None, return as `data`'s type as passed on to the class originally,
            by default None

        Returns
        -------
        np.ndarray | pd.DataFrame
            Multi-hot encoded data.
        """

        columns = self._df.columns
        index = data.index if isinstance(data, pd.DataFrame) else None

        ### Either invert 1d labels (provided as np.ndarray or pd.DataFrame)
        if (isinstance(data, np.ndarray) and data.ndim == 1) or (
            isinstance(data, pd.DataFrame) and list(data.columns) == ["_label_"]
        ):
            values = np.r_[
                [
                    self.id2comb[int(label)]
                    for label in pd.DataFrame(data).astype(int).values
                ]
            ]

        ### Or invert one-hot encoded labels (provided as np.ndarray or pd.DataFrame)
        else:
            values = np.r_[
                [self.onehot2comb[tuple(row)] for row in pd.DataFrame(data).values]
            ]

        return self._return_as(
            pd.DataFrame(values, index=index, columns=columns), astype=astype
        )

    @staticmethod
    def encode_multilabel_name(
        label_names: Sequence[str], multiplicity: Sequence[int] = None
    ) -> str:
        """Format multi-label-names with multiplicity (multiple instances of the same
        class active simultaneously).
        e.g., (["heater", "pump"], [2, 1]) -> "heater*2+pump"
        """
        if multiplicity is None:
            multiplicity = [1] * len(label_names)
        elif len(label_names) != len(multiplicity):
            raise ValueError(
                "If provided, `multiplicity` must have the same length as `label_names`."
            )
        _fmt = lambda col, m: f"{m}*{str(col)}" if m > 1 else str(col)
        return "+".join(
            [_fmt(col, m) for col, m in zip(label_names, multiplicity) if m > 0]
        )

    @staticmethod
    def decode_multilabel_name(
        label_name: str,
    ) -> tuple[Sequence[str], Sequence[int]]:
        """Decode a multi-label name.
        e.g., "heater*2+pump" -> (["heater", "pump"], [2, 1])
        """
        labels_with_multiplicity = label_name.split("+")
        label_names, multiplicity = [], []
        for label in labels_with_multiplicity:
            split_label_and_multiplicity = label.split("*")
            label_names.append(split_label_and_multiplicity[0])
            multiplicity.append(
                1
                if len(split_label_and_multiplicity) == 1
                else split_label_and_multiplicity[1]
            )
        return label_names, multiplicity


