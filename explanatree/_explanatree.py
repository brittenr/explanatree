from typing import Hashable
import pandas as pd

from ._treeframe import TreeFrame


class ExplanaTree:
    def __init__(self, tree_frame: TreeFrame) -> None:
        assert isinstance(tree_frame, TreeFrame)
        self.tree_frame = tree_frame
        self.frame = self.tree_frame.frame

    def get_feature_split_count(self) -> dict[Hashable, int]:
        """Count number of splits on each feature

        Returns:
            dict[str, int]: _description_
        """
        count_dict = self.frame.groupby(["split_feature"])["split_feature"].count().to_dict()

        return count_dict

    def get_feature_split_gain(self) -> dict[Hashable, float]:
        """Sum gain of splits from each feature

        Returns:
            dict[str, float]:
        """
        gain_dict = self.frame.groupby(["split_feature"])["split_gain"].sum().to_dict()

        return gain_dict


if __name__ == "__main__":
    lgbm_df = pd.read_csv("tests/example_data/petal_width_lgbm.csv")

    print(lgbm_df)
    explanatree = ExplanaTree(TreeFrame(lgbm_df))

    print(explanatree.get_feature_split_count())
    print(explanatree.get_feature_split_gain())
