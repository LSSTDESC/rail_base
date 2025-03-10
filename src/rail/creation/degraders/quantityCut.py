"""Degrader that applies a cut to given columns."""

from numbers import Number
from typing import Any

import numpy as np
from ceci.config import StageParameter as Param

from rail.creation.selector import Selector


class QuantityCut(Selector):
    """Degrader that applies a cut to the given columns.

    Note that if a galaxy fails any of the cuts on any one of its columns, that
    galaxy is removed from the sample.
    """

    name = "QuantityCut"
    config_options = Selector.config_options.copy()
    config_options.update(
        cuts=Param(dict, required=True, msg="Cuts to apply"),
    )

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Constructor.

        Performs standard Degrader initialization as well as defining the cuts
        to be applied.
        """
        super().__init__(args, **kwargs)
        self.cuts: dict | None = None
        self.set_cuts(self.config["cuts"])

    def set_cuts(self, cuts: dict) -> None:
        """Defines the cuts to be applied.

        Parameters
        ----------
        cuts : dict
            A dictionary of cuts to make on the data

        Notes
        -----
        The cut keys should be the names of columns you wish to make cuts on.

        The cut values should be either:
        - a number, which is the maximum value. E.g. if the dictionary
        contains "i": 25, then values of i > 25 are cut from the sample.
        - an iterable, which is the range of acceptable values. E.g. if the
        dictionary contains "redshift": (1.5, 2.3), then redshifts outside that
        range are cut from the sample.
        """
        # check that cuts is a dictionary
        if not isinstance(cuts, dict):  # pragma: no cover
            raise TypeError("cuts must be a dictionary.")

        # validate all the cuts and standardize format in dictionary
        self.cuts = {}
        for quantity, cut in cuts.items():
            bad_cut_msg = (
                f"Cut for {quantity} must be a number or an iterable of (min, max)"
            )
            # if single number is provided, save that as the maximum
            if isinstance(cut, Number):
                self.cuts[quantity] = (-np.inf, cut)
            # else, if it's an iterable...
            elif hasattr(cut, "__iter__"):
                # if the iterable is a string or dict, raise error
                if isinstance(cut, (dict, str)):
                    raise TypeError(bad_cut_msg)
                # if the length of the iterable isn't 2, raise error
                if len(cut) != 2:
                    raise ValueError(bad_cut_msg)
                # check that both of the cut values are a number
                for c in cut:
                    if not isinstance(c, Number):
                        raise TypeError(bad_cut_msg)
                # check that the cuts are (min, max)
                if cut[0] > cut[1]:
                    raise ValueError(
                        f"The max for {quantity} must be greater than the min."
                    )
                # if all of these checks passed, save the cuts
                self.cuts[quantity] = (cut[0], cut[1])
            # if the cut isn't a number or iterable, it's the wrong type
            else:
                raise TypeError(bad_cut_msg)

    def _select(self) -> np.ndarray:
        """Applies cuts.

        Notes
        -----
        Gets the input data from the data store under this stage's 'input' tag.

        Puts the data into the data store under this stage's 'output' tag.
        """
        data = self.get_data("input")

        assert self.cuts is not None

        # get overlap of columns from data and columns on which to make cuts
        columns = set(self.cuts.keys()).intersection(data.columns)

        if len(columns) == 0:  # pragma: no cover
            return np.ones(len(data), dtype=int)
        # generate a pandas query from the cuts
        query = [
            f"{col} > {self.cuts[col][0]} & {col} < {self.cuts[col][1]}"
            for col in columns
        ]
        query_st = " & ".join(query)
        out_indices = data.query(query_st).index.values
        out_mask = np.zeros(len(data), dtype=int)
        out_mask[out_indices] = 1
        return out_mask

    def __repr__(self) -> str:  # pragma: no cover
        """Pretty print this object."""
        printMsg = "Degrader that applies the following cuts to a pandas DataFrame:\n"
        printMsg += "{column: (min, max), ...}\n"
        printMsg += self.cuts.__str__()
        return printMsg
