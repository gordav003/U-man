from __future__ import annotations

"""Skupna orodja za ločevanje časovnih vrst v zvezne merilne segmente."""

import pandas as pd


def label_continuous_segments(
    frame: pd.DataFrame,
    *,
    time_column: str = "time",
    maximum_gap: pd.Timedelta | None = None,
    expected_interval: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Vrne urejeno kopijo z zaporedno številko segmenta v stolpcu ``segment``.

    Uporabi se natanko eno pravilo: največja dovoljena vrzel ali strogo
    pričakovani interval. Neveljaven, podvojen ali nazaj obrnjen čas vedno
    začne nov segment.
    """
    if (maximum_gap is None) == (expected_interval is None):
        raise ValueError(
            "Podaj natanko enega od maximum_gap ali expected_interval."
        )

    threshold = maximum_gap if maximum_gap is not None else expected_interval
    if threshold is None or threshold <= pd.Timedelta(0):
        raise ValueError("Časovni prag segmenta mora biti večji od 0.")

    result = (
        frame.dropna(subset=[time_column])
        .sort_values(time_column)
        .reset_index(drop=True)
        .copy()
    )
    gaps = result[time_column].diff()
    if expected_interval is not None:
        starts_segment = gaps.isna() | (gaps != expected_interval)
    else:
        starts_segment = gaps.isna() | (gaps <= pd.Timedelta(0)) | (gaps > threshold)

    result["segment"] = starts_segment.cumsum().astype(int)
    return result
