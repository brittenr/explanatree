from typing import Union

import pandas as pd


def guess_is_df_from_xgb(df: pd.DataFrame) -> bool:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        bool: _description_
    """
    xgb_tree_to_dataframe_cols = [
        "Tree",
        "Node",
        "ID",
        "Feature",
        "Split",
        "Yes",
        "No",
        "Missing",
        "Gain",
        "Cover",
    ]

    has_cols = check_cols(xgb_tree_to_dataframe_cols, df)

    return has_cols


def guess_is_df_from_lgbm(df: pd.DataFrame) -> bool:

    lgbm_tree_to_dataframe_cols = [
        "tree_index",
        "node_depth",
        "node_index",
        "left_child",
        "right_child",
        "parent_index",
        "split_feature",
        "split_gain",
        "threshold",
        "decision_type",
        "missing_direction",
        "missing_type",
        "value",
        "weight",
        "count",
    ]

    has_cols = check_cols(lgbm_tree_to_dataframe_cols, df)

    return has_cols


def check_cols(expected_cols: Union[set, list], df: pd.DataFrame) -> bool:
    assert len(expected_cols) > 0, "Expected cols must have at least one element"
    return set(expected_cols).issubset(set(df.columns))


def parse_lgbm_trees_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df


def parse_xgb_trees_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def parse_unknown_trees_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def derive_source(tree_df):
    if guess_is_df_from_lgbm(tree_df):
        return "lgbm"

    if guess_is_df_from_xgb(tree_df):
        return "xgb"

    return "unknown"
    

def validate_treeframe(df: pd.DataFrame) -> None:
    assert isinstance(df, pd.DataFrame), "Valid TreeFrame should be instance of pandas.DataFrame"


def process_raw_dataframe(raw_dataframe, source) -> pd.DataFrame:
    print(f"Processing provided data frame assuming {source} trees to dataframe export")

    processing_funs = {
        "xgb": parse_xgb_trees_to_dataframe,
        "lgbm": parse_lgbm_trees_to_dataframe,
    }

    output_dataframe = processing_funs.get(source, parse_unknown_trees_to_dataframe)(
        raw_dataframe
    )

    validate_treeframe(output_dataframe)

    return output_dataframe


class TreeFrame:
    def __init__(self, raw_dataframe: pd.DataFrame) -> None:
        self.raw_dataframe = raw_dataframe
        self.source = derive_source(self.raw_dataframe)
        self.frame = process_raw_dataframe(self.source, self.source)