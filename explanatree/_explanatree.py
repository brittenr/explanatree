from typing import Hashable
from abc import ABC

import pandas as pd

from _treeframe import TreeFrame


class Explanatree(ABC):
    def __init__(self, tree_frame: TreeFrame) -> None:
        assert isinstance(tree_frame, TreeFrame)
        self.tree_frame = tree_frame
        self.df = self.tree_frame.df  # duplication for shorthand


class FeatreeImportance(Explanatree):
    def get_feature_split_count(self) -> dict[Hashable, int]:
        """Count number of splits on each feature

        Returns:
            dict[str, int]: _description_
        """
        count_dict = (
            self.df.groupby(["split_feature"])["split_feature"]
            .count()
            .to_dict()  # pylance: ignore
        )

        return count_dict

    def get_feature_split_gain(self) -> dict[Hashable, float]:
        """Sum gain of splits from each feature

        Returns:
            dict[str, float]:
        """
        gain_dict = (
            self.df.groupby(["split_feature"])["split_gain"].sum().to_dict()
        )  # pylance: ignore

        return gain_dict


class TreeDependencePlots(Explanatree):
    pass


if __name__ == "__main__":
    lgbm_df = pd.read_csv("tests/example_data/petal_width_lgbm.csv")

    print(lgbm_df)
    explanatree = FeatreeImportance(TreeFrame(lgbm_df))

    print(explanatree.get_feature_split_count())
    print(explanatree.get_feature_split_gain())
