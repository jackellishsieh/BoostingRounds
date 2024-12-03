import pandas as pd
import numpy as np

from typing import Dict, Callable, Tuple
import sklearn.model_selection
from pathlib import Path
import outlier_generation_techniques
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score

Row = pd.core.series.Series

'''
Given a dataframe and a dictionary associating names to type boolean functions, checks by assertion that the columns exist AND the types match the dictionary

For example,    assert_column_types(df, {"x1": pd.api.types.is_integer_dtype}) 
asserts that df has a column named "x1" of type integer
'''
def assert_column_types(df: pd.DataFrame, column_type_functions: Dict[str, Callable[[Row], bool]]) -> None:
    for column_name, column_type_function in column_type_functions.items():
        assert column_name in df.columns, f"Column {column_name} is not in the DataFrame"
        assert column_type_function(df[column_name]), f"Column {column_name} does not satisfy {column_type_function.__name__}"

'''
Given a dataframe and the column name of an optional parameter, returns the column value if it exists and is not NA, and returns default otherwise
'''
def get_optional_argument(row: Row, column_name: str, default: str):
    argument_provided: bool = column_name in row.keys() and pd.notna(row[column_name])     # can't check type, lost in row form
    return row[column_name] if argument_provided else default


'''
Given a dataframe and a column name of filenames, return a dictionary associating filenames to dataframes
'''
def get_filename_TO_dataframe(df: pd.DataFrame, column_name: str) -> Dict[str, pd.DataFrame]:
    return {dataset_filename: pd.read_csv(dataset_filename, index_col=0) for dataset_filename in df[column_name].unique()}

'''
Given a dataframe with a labels column,...

get_X       return without the labels column
get_y       return just the labels column
get_X_y     return a tuple of both
'''
def get_X(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop("label", axis = 1)

def get_y(df: pd.DataFrame) -> pd.core.series.Series:
    return df["label"]

def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.core.series.Series]:
    return (get_X(df), get_y(df))

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
    dataset_filename_TO_dataframe = get_filename_TO_dataframe(df, "dataset_filename")

    # Given a row, create the split, capture in a file, and return the train and test dataset filenames
    # If the filenames already exist, do nothing
    def create_dataset_split(row: pd.core.series.Series) -> Tuple[str, str]:        
        dataset_filename = row["dataset_filename"]
        test_proportion = row["test_proportion"]
        random_state = row["dataset_split_random_state"]

        # Pathname is just the name of the original dataset file without the .csv
        parent_pathname = dataset_filename.removesuffix(".csv")
        Path(parent_pathname).mkdir(parents=True, exist_ok=True)            # Creates the given directory IF it doesn't already exist

        train_dataset_filename = f"{parent_pathname}/train_dataset-test_proportion={test_proportion}-random_state={random_state}.csv"
        test_dataset_filename = f"{parent_pathname}/test_dataset-test_proportion={test_proportion}-random_state={random_state}.csv"

        # Only create a new split if the files don't already exist
        if not (Path(train_dataset_filename).exists() and Path(test_dataset_filename).exists()):
            dataset = dataset_filename_TO_dataframe[dataset_filename]
            train_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, test_size = test_proportion, random_state = random_state)
            train_dataset.to_csv(train_dataset_filename)
            test_dataset.to_csv(test_dataset_filename)

        return train_dataset_filename, test_dataset_filename
    
    # Apply to each row
    df[["train_dataset_filename", "test_dataset_filename"]] = df.apply(create_dataset_split, axis=1, result_type="expand")
    return

'''
Input: a pandas dataframe containing columns...
- "dataset_filename" (str)          (should be a filename to a csv of a dataframe with proper label as last column) -> uses the ENTIRE dataset
- "outlier_method" (str)            (doesn't do anything yet)
- "alpha" (float)
- "epsilon" (float)
- "outlier_proportion" (float)
- "outlier_random_state" (int)

Output: None
Updates the provides pandas dataframe in-place to add one new column:
- "outlier_dataset_filename" (str)
and creates the file accordingly in a subfolder with the same filename
'''
def create_outlier_sets(df: pd.DataFrame) -> None:
    # Assert that the input columns exist and are of the right type
    assert_column_types(df, {
        "dataset_filename": pd.api.types.is_string_dtype,
        "outlier_method": pd.api.types.is_string_dtype,
        "alpha": pd.api.types.is_float_dtype,
        "epsilon": pd.api.types.is_float_dtype,
        "outlier_proportion": pd.api.types.is_float_dtype,
        "outlier_random_state": pd.api.types.is_integer_dtype
    })

    assert (df["outlier_method"] == "infeasExamRandomLabel").all()  # only supported method currently

    # First, memoize reading in each input filename so we don't have to repeat so many times
    dataset_filename_TO_dataframe = get_filename_TO_dataframe(df, "dataset_filename")

    # Given a row, create the outlier set, capture in a file, and return the outlier set filename
    def create_outlier_set(row: pd.core.series.Series) -> Tuple[str]:        
        dataset_filename = row["dataset_filename"]
        alpha = row["alpha"]
        epsilon = row["epsilon"]
        outlier_proportion = row["outlier_proportion"]
        random_state = row["outlier_random_state"]

        # Pathname is just the name of the original dataset file without the .csv
        parent_pathname = dataset_filename.removesuffix(".csv")
        Path(parent_pathname).mkdir(parents=True, exist_ok=True)            # Creates the given directory IF it doesn't already exist
        outlier_set_filename = f"{parent_pathname}/outlier_dataset-alpha={alpha}-epsilon={epsilon}-outlier_proportion={outlier_proportion}-random_state={random_state}.csv"

        if not Path(outlier_set_filename).exists():
            dataset = dataset_filename_TO_dataframe[dataset_filename]
            outlier_set = outlier_generation_techniques.infeasExamRandomLabel(dataset, outlier_proportion, alpha, epsilon, random_state)
            outlier_set.to_csv(outlier_set_filename)

        return (outlier_set_filename,)


    # Apply to each row
    df[["outlier_dataset_filename"]] = df.apply(create_outlier_set, axis=1, result_type="expand")
    return


'''
Helper function: takes in a classifier and three dataframes, and returns the 
'''

'''
Input: a pandas dataframe containing columns...
- "train_dataset_filename" (str)    (should be a filename to a csv of a dataframe with proper label as last column)
- "test_dataset_filename" (str)     (should be a filename to a csv of a dataframe with proper label as last column)
- "outlier_dataset_filename" (str)  (should be a filename to a csv of a dataframe with proper label as last column)

- "classifier_type"  (str)
    - if "classifier_type" = "LogisticRegression", then can optionally take parameters...
        - "LogisticRegression.penalty" (Object) = "l2", None
    - if "classifier_type" = "GradientBoostingClassifier", then can optionally take parameters...
        - "GradientBoostingClassifier.n_estimators" (int)
        - "GradientBoostingClassifier.loss" (str)
        - "GradientBoostingClassifier.max_depth" (int)
        - "GradientBoostingClassifier.learning_rate" (float)       higher value means higher "regularization"

radientBoostingClassifier(*, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

Output: None
Updates the provides pandas dataframe in-place to add one new column:
- "final_log_loss_train_dataset" (float)
- "final_log_loss_contaminated_train_dataset" (float)
- "final_log_loss_test_dataset" (float)
- "loss_data_filename" (str or pd.NA if no such file)

- "final_accuracy_train_dataset" (float)
- "final_accuracy_contaminated_train_dataset" (float)
- "final_accuracy_test_dataset" (float)
- "accuracy_data_filename" (str or pd.NA if no such file)

and creates the file accordingly in a subfolder with the same filename (stores  pandas dataframe of train loss, test loss, contaminated train loss and (if possible) iteration number)
'''
def evaluate_classifiers(df: pd.DataFrame) -> None:
    # Assert that necessary input columns exist and are of the right type
    assert_column_types(df, {
        "train_dataset_filename": pd.api.types.is_string_dtype,
        "test_dataset_filename": pd.api.types.is_string_dtype,
        "outlier_dataset_filename": pd.api.types.is_string_dtype,
        "classifier_type": pd.api.types.is_string_dtype
    })
    
    # First, memoize reading in each input filename so we don't have to repeat so many times
    train_dataset_filename_TO_dataframe = get_filename_TO_dataframe(df, "train_dataset_filename")
    test_dataset_filename_TO_dataframe = get_filename_TO_dataframe(df, "test_dataset_filename")
    outlier_dataset_filename_TO_dataframe = get_filename_TO_dataframe(df, "outlier_dataset_filename")
    
    rows_completed = 0
    total_rows = len(df)

    def evaluate_classifier(row: pd.core.series.Series) -> Tuple[float, float, float, str, float, float, float, str]:
        # Keep track of progress
        nonlocal rows_completed
        rows_completed += 1
        print(f"starting row {rows_completed} of {total_rows}")

        # Extract the arguments
        train_dataset_filename = row["train_dataset_filename"]
        test_dataset_filename = row["test_dataset_filename"]
        outlier_dataset_filename = row["outlier_dataset_filename"]
        classifier_type = row["classifier_type"]
        
        train_dataframe = train_dataset_filename_TO_dataframe[train_dataset_filename]
        test_dataframe = test_dataset_filename_TO_dataframe[test_dataset_filename]
        outlier_dataframe = outlier_dataset_filename_TO_dataframe[outlier_dataset_filename].astype(train_dataframe.dtypes.to_dict())    # to prevent empty from being object type

        # Construct contaminated dataframe
        contaminated_train_dataframe = pd.concat([train_dataframe, outlier_dataframe], axis=0, ignore_index=True)

        # If the classifier is LogisticRegression, then only evaluate final (no staged possible)
        if classifier_type == "LogisticRegression":
            # Set the penalty to the function's default, unless otherwise specified by the row
            penalty = get_optional_argument(row, "LogisticRegression.penalty", "l2")

            # Create logistic regression classifier and fit to contaminated dataset
            clf = LogisticRegression(penalty = penalty)
            clf.fit(get_X(contaminated_train_dataframe), get_y(contaminated_train_dataframe))

            # Evaluate the final log losses
            final_log_loss_train_dataset, final_log_loss_contaminated_train_dataset, final_log_loss_test_dataset = [
                log_loss(get_y(dataframe), clf.predict_proba(get_X(dataframe))) 
                for dataframe in [train_dataframe, contaminated_train_dataframe, test_dataframe]
                ]
            loss_data_filename = pd.NA      # don't save to a file, unnecessary since only final results

            # Evaluate the final accuracies
            final_accuracy_train_dataset, final_accuracy_contaminated_train_dataset, final_accuracy_test_dataset = [
                accuracy_score(get_y(dataframe), clf.predict(get_X(dataframe)))
                for dataframe in [train_dataframe, contaminated_train_dataframe, test_dataframe]
            ]
            accuracy_data_filename = pd.NA      # don't save to a file, unnecessary since only final results

            return (final_log_loss_train_dataset, final_log_loss_contaminated_train_dataset, final_log_loss_test_dataset, loss_data_filename,
                    final_accuracy_train_dataset, final_accuracy_contaminated_train_dataset, final_accuracy_test_dataset, accuracy_data_filename)
        
        elif classifier_type == "GradientBoostingClassifier":
            # Get optional parameters
            n_estimators = get_optional_argument(row, "GradientBoostingClassifier.n_estimators", 100)
            loss = get_optional_argument(row, "GradientBoostingClassifier.loss", "log_loss")
            max_depth = get_optional_argument(row, "GradientBoostingClassifier.max_depth", 3)
            learning_rate = get_optional_argument(row, "GradientBoostingClassifier.learning_rate", 0.1)

            # Create the classifier and train on the contaminated dataset
            clf = GradientBoostingClassifier(n_estimators=n_estimators, loss=loss, max_depth=max_depth, learning_rate=learning_rate)
            clf.fit(get_X(contaminated_train_dataframe), get_y(contaminated_train_dataframe))

            # Compute the log losses and the accuracies at each stage
            loss_data_dict = {}
            accuracy_data_dict = {}

            # Compute log loss and accuracy for each dataset and stage
            for loss_column, accuracy_column, dataset in zip(
                ["log_loss_train_dataset", "log_loss_contaminated_train_dataset", "log_loss_test_dataset"],
                ["accuracy_train_dataset", "accuracy_contaminated_train_dataset", "accuracy_test_dataset"],
                [train_dataframe, contaminated_train_dataframe, test_dataframe]
            ):
                X, y_true = get_X(dataset), get_y(dataset)
                loss_data_dict[loss_column] =           [log_loss(y_true, y_pred_prob) for y_pred_prob in clf.staged_predict_proba(X)]
                accuracy_data_dict[accuracy_column] =   [accuracy_score(y_true, y_pred) for y_pred in clf.staged_predict(X)]

            # Convert each dictionary to a DataFrame
            loss_dataframe = pd.DataFrame(loss_data_dict)
            final_log_loss_train_dataset, final_log_loss_contaminated_train_dataset, final_log_loss_test_dataset = loss_dataframe.iloc[-1][loss_data_dict.keys()]   # guaranteed ordered

            accuracy_dataframe = pd.DataFrame(accuracy_data_dict)
            final_accuracy_train_dataset, final_accuracy_contaminated_train_dataset, final_accuracy_test_dataset = accuracy_dataframe.iloc[-1][accuracy_data_dict.keys()]   # guaranteed ordered
            
            # Save dataframes to files, under name of OUTLIER dataset
            parent_pathname = outlier_dataset_filename.removesuffix(".csv")
            Path(parent_pathname).mkdir(parents=True, exist_ok=True)            # Creates the given directory IF it doesn't already exist

            loss_data_filename = f"{parent_pathname}/loss_data-classifier_type='{classifier_type}'-n_estimators={n_estimators}-loss='{loss}'-max_depth={max_depth}-learning_rate={learning_rate}.csv"
            loss_dataframe.to_csv(loss_data_filename)

            accuracy_data_filename = f"{parent_pathname}/accuracy_data-classifier_type='{classifier_type}'-n_estimators={n_estimators}-loss='{loss}'-max_depth={max_depth}-learning_rate={learning_rate}.csv"
            accuracy_dataframe.to_csv(accuracy_data_filename)

            # Return everything
            return (final_log_loss_train_dataset, final_log_loss_contaminated_train_dataset, final_log_loss_test_dataset, loss_data_filename,
                    final_accuracy_train_dataset, final_accuracy_contaminated_train_dataset, final_accuracy_test_dataset, accuracy_data_filename)

        else:
            assert False, f"Improper classifier type '{classifier_type}' provided"
        
    produced_columns = ["final_log_loss_train_dataset", "final_log_loss_contaminated_train_dataset", "final_log_loss_test_dataset", "loss_data_filename",
                        "final_accuracy_train_dataset", "final_accuracy_contaminated_train_dataset", "final_accuracy_test_dataset", "accuracy_data_filename"]
    
    # If one of the produced columns doesn't exist, process all
    if not all(col in df.columns for col in produced_columns):
        df[produced_columns] = df.apply(evaluate_classifier, axis=1, result_type="expand")
    else:
        # Otherwise, only execute on rows with na in the first column (not all, since last column might be na)
        df.loc[df["final_log_loss_train_dataset"].isna(), produced_columns] = df.loc[df["final_log_loss_train_dataset"].isna()].apply(evaluate_classifier, axis=1, result_type="expand")

    return