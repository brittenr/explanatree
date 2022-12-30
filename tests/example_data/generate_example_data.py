import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm
from pathlib import Path


def main() -> None:
    target = "petal_width"
    output_path = Path("tests/example_data")
    output_path.mkdir(exist_ok=True)

    # prepare data
    data: pd.DataFrame = sns.load_dataset("iris")  # type: ignore
    data["species"] = data["species"].astype("category")

    # model xgboost
    dtrain = xgb.DMatrix(
        data.drop([target, "species"], axis=1),
        label=data[target],
        enable_categorical=True,
    )

    xgb_model = xgb.train(params={"random_state": 0}, dtrain=dtrain)

    xgb_model.trees_to_dataframe().to_csv(output_path / f"{target}_xgb.csv")

    # model lgbm
    lgbm_train = lgbm.Dataset(data.drop(target, axis=1), label=data[target])

    lgbm_model = lgbm.train(params={"random_state": 0}, train_set=lgbm_train)

    lgbm_model.trees_to_dataframe().to_csv(output_path / f"{target}_lgbm.csv")


if __name__ == "__main__":
    main()
