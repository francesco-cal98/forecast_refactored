import pandas as pd
from backend.data.dataset import SalesDataset

def test_dataset_loading_and_processing():
    # Usa un piccolo DataFrame finto (puoi adattarlo a un CSV se preferisci)
    df = pd.DataFrame({
        "PERIOD_TAG": pd.date_range("2023-01-01", periods=10, freq="W"),
        "Sellout": range(10),
        "product_group": ["A"] * 10,
        "type_promo_1": [0, 1] * 5,
        "type_promo_2": [1, 0] * 5,
        "customer_name": ["X"] * 10
    })
    df.to_csv("tests/temp_test.csv", index=False)

    dataset = SalesDataset(
        csv_file="tests/temp_test.csv",
        separator=",",
        date_column="PERIOD_TAG",
        target_column="Sellout",
        nr_horizon=1,
        prepare_json="config/prepare_dataset.json",
        column_selection_fp="config/column_selection.json",
        product_var="product_group"
    )

    dataset.process_data(product="A", train_params={
        "train_len": "6w",
        "test_len": "2w",
        "boundary": None
    })

    assert dataset.X_train is not None
    assert dataset.y_train is not None
    assert dataset.X_test is not None
    assert dataset.y_test is not None
    assert len(dataset.X_train) > 0
