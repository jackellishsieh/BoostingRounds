import pandas as pd
from typing import Dict, Callable, Tuple
import sklearn.model_selection
from pathlib import Path

'''
Given a dataframe and a dictionary associating names to type boolean functions, checks by assertion that the columns exist AND the types match the dictionary

For example,    assert_column_types(df, {"x1": pd.api.types.is_integer_dtype}) 
asserts that df has a column named "x1" of type integer
'''
def assert_column_types(df: pd.DataFrame, column_type_functions: Dict[str, Callable[[pd.core.series.Series], bool]]) -> None:
    for column_name, column_type_function in column_type_functions.items():
        assert column_name in df.columns, f"Column {column_name} is not in the DataFrame"
        assert column_type_function(df[column_name]), f"Column {column_name} does not satisfy {column_type_function.__name__}"

'''
Input: a pandas dataframe containing columns...
- "dataset_filename" (str)          (should be a filename to a csv of a dataframe with proper label as last column)
- "test_proportion" (float)
- "dataset_split_random_state" (int)
Output: None

Updates the provides pandas dataframe in-place to add two new columns:
- "test_dataset_filename" (str)
- "train_dataset_filename" (str)
and creates the files accordingly in a subfolder with the same filename
'''
def create_dataset_splits(df: pd.DataFrame) -> None:
    # Assert that the input columns exist and are of the right type
    assert_column_types(df, {
        "dataset_filename": pd.api.types.is_string_dtype,
        "test_proportion": pd.api.types.is_float_dtype,
        "dataset_split_random_state": pd.api.types.is_integer_dtype
    })

    # First, memoize reading in each input filename so we don't have to repeat so many times
    # print("unique = ", df["dataset_filename"].unique())
    dataset_filename_TO_dataframe = {dataset_filename: pd.read_csv(dataset_filename, index_col=0) for dataset_filename in df["dataset_filename"].unique()}

    # Given a row, create the split, capture in a file, and return the train and test dataset filenames
    def create_dataset_split(row: pd.core.series.Series) -> Tuple[str, str]:        
        dataset_filename = row["dataset_filename"]
        test_proportion = row["test_proportion"]
        random_state = row["dataset_split_random_state"]

        dataset = dataset_filename_TO_dataframe[dataset_filename]

        # Pathname is just the name of the original dataset file without the .csv
        parent_pathname = dataset_filename.removesuffix(".csv")
        Path(parent_pathname).mkdir(parents=True, exist_ok=True)            # Creates the given directory IF it doesn't already exist

        train_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, test_size = test_proportion, random_state = random_state)
        train_dataset_filename = f"{parent_pathname}/train-random_state={random_state}.csv"
        test_dataset_filename = f"{parent_pathname}/test-random_state={random_state}.csv"
        train_dataset.to_csv(train_dataset_filename)
        test_dataset.to_csv(test_dataset_filename)

        return train_dataset_filename, test_dataset_filename

        # row["train_dataset_filename"] = train_dataset_filename
        # row["test_dataset_filename"] = test_dataset_filename
        
        return row
    
    # Apply to each row
    df[["train_dataset_filename", "test_dataset_filename"]] = df.apply(create_dataset_split, axis=1, result_type="expand")
    return

