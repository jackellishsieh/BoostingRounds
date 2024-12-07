import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Transforms a column name to a spaced, capitalized name
def dress(s: str):
    return s.replace("_", " ").title()


# Compares different max depth's with the same proportion AND logistic regression
def plot_performance_comparison_by_outlier_proportion(GBDT_df, log_reg_df, performance_type="log_loss", legend=True, ax=None):
    # GRADIENT BOOSTING DATA
    # Sort the dataframe by max_depth (ascending) and then by learning_rate (ascending)
    # The first element is the least flexible learning model, while the last element is the most flexible
    GBDT_df = GBDT_df.sort_values(
        by=['GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate']
    ).reset_index(drop=True)
    
    # Create a combined column for learning_rate and max_depth
    GBDT_df['Learning Parameters'] = "d=" + GBDT_df['GradientBoostingClassifier.max_depth'].astype(str) +\
                                         ",v=" + GBDT_df['GradientBoostingClassifier.learning_rate'].astype(str)
    
    num_lines = GBDT_df['Learning Parameters'].nunique()
    color_palette = sns.color_palette("plasma", num_lines)
    # color_palette.reverse()         # warm is lower variability, cool is higher variability
    color_map = {name: color_palette[i] for i, name in enumerate(GBDT_df['Learning Parameters'].unique())}

    # Melt the dataframe for easier plotting with seaborn
    GBDT_df_melted = GBDT_df.melt(
        id_vars=['outlier_proportion', 'Learning Parameters'],
        value_vars=[f'final_{performance_type}_contaminated_train_dataset', f'final_{performance_type}_test_dataset'],
        var_name='Dataset',
        value_name=dress(performance_type)
    )
    
    # Rename the Dataset column for better readability
    GBDT_df_melted['Dataset'] = GBDT_df_melted['Dataset'].replace({
        f'final_{performance_type}_contaminated_train_dataset': 'Train (contaminated)',
        f'final_{performance_type}_test_dataset': 'Test'
    })

    # LOGISTIC REGRESSION DATA
    log_reg_df['Learning Parameters'] = 'Logistic Regression'
    log_reg_melted = log_reg_df.melt(
        id_vars=['outlier_proportion', 'Learning Parameters'],
        value_vars=[f'final_{performance_type}_contaminated_train_dataset', f'final_{performance_type}_test_dataset'],
        var_name='Dataset',
        value_name=dress(performance_type)
    )
    log_reg_melted['Dataset'] = log_reg_melted['Dataset'].replace({
        'outlier_proportion': 'High Leverage Proportion',
        f'final_{performance_type}_contaminated_train_dataset': 'Train (contaminated)',
        f'final_{performance_type}_test_dataset': 'Test'
    })


    # Combine both dataframes for plotting
    combined_df = pd.concat([GBDT_df_melted, log_reg_melted], ignore_index=True)

    # If no axes, create
    if ax is None:
        ax = plt.gca()

    # Draw the plot on the axes
    sns.lineplot(
        data=combined_df,
        x= 'outlier_proportion',
        y= dress(performance_type),

        errorbar='sd',  # Confidence interval

        # Color by learning parameter combinations, in order of increasing model flexibility
        hue='Learning Parameters',
        palette={**color_map, 'Logistic Regression': 'green'},  # different color for Logistic Regression

        # Solid line for test, dashed line for train (contaminated)
        style='Dataset',
        style_order=["Test", "Train (contaminated)"],
        legend="full",
        ax=ax
    )
    
    # Add labels and legend
    ax.set_title(f'{dress(performance_type)} vs High Leverage Proportion')
    ax.set_xlabel("High Leverage Proportion (p)")
    ax.set_ylabel(dress(performance_type))
    ax.legend(
        # title='Learning Parameters / Dataset',
        loc='center left',
        bbox_to_anchor=(1, 0.5),
    ).set_visible(legend)
    ax.grid(True)
    
    # Return the figure
    return ax


# Combines GradientBoostingClassifier.learning_rate and GradientBoostingClassifier.max_depth
def plot_performance_by_outlier_proportion(GBDT_df, performance_type="log_loss", legend=True, ax=None):
    """
    Plots curves with confidence intervals for (contaminated) train and test log loss, 
    with respect to outlier proportion, and colors by both learning_rate and max_depth.
    
    Parameters:
        dataframe (pd.DataFrame): DataFrame containing columns:
            'outlier_proportion', 'GradientBoostingClassifier.learning_rate', 
            'GradientBoostingClassifier.max_depth',
            'final_log_loss_train_dataset', 'final_log_loss_test_dataset'.
    
    Returns:
        matplotlib.figure.Figure: The resulting plot as a figure object.
    """
    # GRADIENT BOOSTING DATA
    # Sort the dataframe by max_depth (ascending) and then by learning_rate (ascending)
    # The first element is the least flexible learning model, while the last element is the most flexible
    GBDT_df = GBDT_df.sort_values(
        by=['GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate']
    ).reset_index(drop=True)
    
    # Create a combined column for learning_rate and max_depth
    GBDT_df['Learning Parameters'] = "d=" + GBDT_df['GradientBoostingClassifier.max_depth'].astype(str) +\
                                         ",v=" + GBDT_df['GradientBoostingClassifier.learning_rate'].astype(str)
    
    num_lines = GBDT_df['Learning Parameters'].nunique()
    color_palette = sns.color_palette("plasma", num_lines)
    # color_palette.reverse()         # warm is lower variability, cool is higher variability
    color_map = {name: color_palette[i] for i, name in enumerate(GBDT_df['Learning Parameters'].unique())}

    # Melt the dataframe for easier plotting with seaborn
    df_melted = GBDT_df.melt(
        id_vars=['outlier_proportion', 'Learning Parameters'],
        value_vars=[f'final_{performance_type}_contaminated_train_dataset', f'final_{performance_type}_test_dataset'],
        var_name='Dataset',
        value_name=dress(performance_type)
    )
    
    # Rename the Dataset column for better readability
    df_melted['Dataset'] = df_melted['Dataset'].replace({
        'outlier_proportion': 'High Leverage Proportion',
        f'final_{performance_type}_contaminated_train_dataset': 'Train (contaminated)',
        f'final_{performance_type}_test_dataset': 'Test'
    })
    
    # If no axes, create
    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        data=df_melted,
        x= 'outlier_proportion',
        y= dress(performance_type),

        errorbar='sd',  # Confidence interval

        # Color by learning parameter combinations, in order of increasing model flexibility
        hue='Learning Parameters',
        palette=color_map,

        # Solid line for test, dashed line for train (contaminated)
        style='Dataset',
        style_order=["Test", "Train (contaminated)"],
        legend="full",
        ax = ax
    )
    
    # Add labels and legend
    ax.set_title(f'{dress(performance_type)} vs High Leverage Proportion')
    ax.set_xlabel("High Leverage Proportion (p)")
    ax.set_ylabel(dress(performance_type))
    ax.legend(
        # title='Learning Parameters / Dataset',
        loc='center left',
        bbox_to_anchor=(1, 0.5),
    ).set_visible(legend)
    ax.grid(True)
    
    # Return the figure
    return ax


def plot_performance_by_iteration(GBDT_df, max_depth: int, learning_rate: float, performance_type="log_loss", legend = True, ax = None):
    """
    Plots curves with confidence intervals for (contaminated) train and test log loss, 
    with respect to boosting number, and colors by outlier proportion.
    
    Parameters:
            dataframe (pd.DataFrame): DataFrame containing columns:
                'outlier_proportion', 'GradientBoostingClassifier.learning_rate', 
                'GradientBoostingClassifier.max_depth',
                'final_log_loss_train_dataset', 'final_log_loss_test_dataset'.    
    
    Returns:
        matplotlib.figure.Figure: The resulting plot as a figure object.
    """
    GBDT_df = GBDT_df.loc[(GBDT_df["GradientBoostingClassifier.max_depth"] == max_depth) \
                            & (GBDT_df["GradientBoostingClassifier.learning_rate"] == learning_rate)]

    combined_data = []
    
    for _, row in GBDT_df.iterrows():
        outlier_proportion = row["outlier_proportion"]
        loss_data = pd.read_csv(row[f"{performance_type}_data_filename"])  # Adjust if files are stored differently
        loss_data["iterations"] = loss_data.index
        loss_data["High Leverage Proportion (p)"] = outlier_proportion
        combined_data.append(loss_data)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Melt the data for easier plotting with seaborn
    df_melted = combined_df.melt(
        id_vars=["iterations", "High Leverage Proportion (p)"],
        value_vars=[f'{performance_type}_contaminated_train_dataset', f'{performance_type}_test_dataset'],
        var_name="Dataset",
        value_name=dress(performance_type)
    )

    # Rename the Dataset column for better readability
    df_melted['Dataset'] = df_melted['Dataset'].replace({
        f'{performance_type}_contaminated_train_dataset': 'Train (contaminated)',
        f'{performance_type}_test_dataset': 'Test'
    })
    
    # If no axes, create
    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        data=df_melted,
        x= 'iterations',
        y= dress(performance_type),

        errorbar='sd',  # Confidence interval

        # Color by learning parameter combinations, in order of increasing model flexibility
        hue='High Leverage Proportion (p)',
        palette="viridis",

        # Solid line for test, dashed line for train (contaminated)
        style='Dataset',
        style_order=["Test", "Train (contaminated)"],
        legend="full",
        ax = ax
    )
    
    ax.set_title(f'{dress(performance_type)} vs Boosting Rounds (d={max_depth},v={learning_rate})')
    ax.set_xlabel("Boosting Rounds (B)")
    ax.set_ylabel(dress(performance_type))
    ax.legend(
        # title='Outlier Proportion / Dataset',
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    ).set_visible(legend)
    ax.grid(True)

    return ax

# Computes a table of performance differences
def create_diff_dataframe(summary_df, outlier_proportion1 = 0.1, outlier_proportion2 = 0):
    y_columns = [
       'mean_log_loss_train_dataset',
       'mean_log_loss_contaminated_train_dataset',
       'mean_log_loss_test_dataset', 'mean_accuracy_train_dataset',
       'mean_accuracy_contaminated_train_dataset',
       'mean_accuracy_test_dataset'
    ]
    
    # Filter the rows for outlier_proportion == 0.1 and outlier_proportion == 0
    df1 = summary_df[summary_df['outlier_proportion'] == outlier_proportion1]
    df2 = summary_df[summary_df['outlier_proportion'] == outlier_proportion2]
    
    # Merge the two dataframes by the boosting rounds and learning parameters
    merged_df = pd.merge(df1, df2, on=['boosting_round', 'GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate'], suffixes=('_1', '_2'))
    
    for col in y_columns:
        diff_col_name = f'diff_{col}'
        merged_df[diff_col_name] = merged_df[f'{col}_1'] - merged_df[f'{col}_2']
    
    # Create a final dataframe
    result_df = merged_df[['boosting_round', 'GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate'] + [f'diff_{col}' for col in y_columns]]
    
    return result_df


# Computes a table of performance difference
def create_difference_dataframe(summary_df, outlier_proportion1 = 0.1, outlier_proportion2 = 0):
    y_columns = [
       'mean_log_loss_train_dataset',
       'mean_log_loss_contaminated_train_dataset',
       'mean_log_loss_test_dataset', 'mean_accuracy_train_dataset',
       'mean_accuracy_contaminated_train_dataset',
       'mean_accuracy_test_dataset'
    ]
    
    # Filter the rows for outlier_proportion == 0.1 and outlier_proportion == 0
    df1 = summary_df[summary_df['outlier_proportion'] == outlier_proportion1]
    df2 = summary_df[summary_df['outlier_proportion'] == outlier_proportion2]
    
    # Merge the two dataframes by the boosting rounds and learning parameters
    merged_df = pd.merge(df1, df2, on=['boosting_round', 'GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate'], suffixes=('_1', '_2'))
    
    for col in y_columns:
        diff_col_name = f'diff_{col}'
        merged_df[diff_col_name] = merged_df[f'{col}_1'] - merged_df[f'{col}_2']
    
    # Create a final dataframe
    result_df = merged_df[['boosting_round', 'GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate'] + [f'diff_{col}' for col in y_columns]]
    
    return result_df

# Plots the difference in performance plots between for parameter combinations
# Combines GradientBoostingClassifier.learning_rate and GradientBoostingClassifier.max_depth
def plot_performance_difference_by_iteration(summary_dataframe, performance_type="log_loss", train=True, test=True, legend=True, ax=None, outlier_proportion1 = 0.1, outlier_propirtion2=0):
    """
    Plots curves with confidence intervals for (contaminated) train and test log loss, 
    with respect to outlier proportion, and colors by both learning_rate and max_depth.
    
    Parameters:
        dataframe (pd.DataFrame): DataFrame containing columns:
            'outlier_proportion', 'GradientBoostingClassifier.learning_rate', 
            'GradientBoostingClassifier.max_depth',
            'final_log_loss_train_dataset', 'final_log_loss_test_dataset'.
    
    Returns:
        matplotlib.figure.Figure: The resulting plot as a figure object.
    """
    # Sort the dataframe by max_depth (ascending) and then by learning_rate (ascending)
    # The first element is the least flexible learning model, while the last element is the most flexible
    # result_df = create_diff_dataframe(summary_dataframe, outlier_proportion1 = 0.1, outlier_proportion2 = 0)
    result_df = create_difference_dataframe(summary_dataframe, outlier_proportion1 = outlier_proportion1, outlier_proportion2 = outlier_propirtion2)

    result_df = result_df.sort_values(
        by=['GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate']
    ).reset_index(drop=True)
    
    # Create a combined column for learning_rate and max_depth
    result_df['Learning Parameters'] = "d=" + result_df['GradientBoostingClassifier.max_depth'].astype(str) +\
                                         ",v=" + result_df['GradientBoostingClassifier.learning_rate'].astype(str)
    
    num_lines = result_df['Learning Parameters'].nunique()
    color_palette = sns.color_palette("plasma", num_lines)
    # color_palette.reverse()         # warm is lower variability, cool is higher variability
    color_map = {name: color_palette[i] for i, name in enumerate(result_df['Learning Parameters'].unique())}

    # Melt the dataframe for easier plotting with seaborn
    value_vars = []
    if train:
        value_vars.append(f'diff_mean_{performance_type}_contaminated_train_dataset',)
    if test:
        value_vars.append(f'diff_mean_{performance_type}_test_dataset')

    df_melted = result_df.melt(
        id_vars=['boosting_round', 'Learning Parameters'],
        value_vars=value_vars,
        var_name='Dataset',
        value_name=dress(performance_type)
    )
    
    # Rename the Dataset column for better readability
    df_melted['Dataset'] = df_melted['Dataset'].replace({
        f'diff_mean_{performance_type}_contaminated_train_dataset': 'Train (contaminated)',
        f'diff_mean_{performance_type}_test_dataset': 'Test'
    })
    
    # If no axes, create
    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        data=df_melted,
        x= 'boosting_round',
        y= dress(performance_type),

        # Color by learning parameter combinations, in order of increasing model flexibility
        hue='Learning Parameters',
        palette=color_map,

        # Solid line for test, dashed line for train (contaminated)
        style='Dataset',
        style_order=["Test", "Train (contaminated)"],
        legend="full",
        ax = ax
    )
    
    # Add labels and legend
    # ax.set_title(f'{dress(performance_type)} difference (p={outlier_proportion1} vs. p={outlier_propirtion2}) vs Boosting Rounds')
    ax.set_xlabel("Boosting Rounds (B)")
    ax.set_ylabel(dress(performance_type))
    ax.legend(
        title='Learning Parameters / Dataset',
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    ).set_visible(legend)
    ax.grid(True)
    
    # Return the figure
    return ax


# # Plots all parameter combinations simultaneously. Plots are distinguished by color only for outlier proportions.
# Does not plot as expected: plots every parameter combination independently
# def plot_all_performance_by_iteration(summary_dataframe, performance_type = "log_loss"):
#     '''
#     Requires a SUMMARY dataframe
#     '''
#      # Melt the dataframe for easier plotting with seaborn
#     df_melted = summary_dataframe.melt(
#         id_vars=['boosting_round', 'outlier_proportion', 'GradientBoostingClassifier.max_depth', 'GradientBoostingClassifier.learning_rate'],
#         # value_vars=[f'mean_{performance_type}_contaminated_train_dataset', f'mean_{performance_type}_test_dataset'],
#         value_vars=[f'mean_{performance_type}_test_dataset'],
#         var_name='Dataset',
#         value_name= dress(performance_type)
#     )
    
#     # Rename the Dataset column for better readability
#     df_melted['Dataset'] = df_melted['Dataset'].replace({
#         # f'mean_{performance_type}_contaminated_train_dataset': 'Train (Contaminated)',
#         f'mean_{performance_type}_test_dataset': 'Test'
#     })
    
#     # Create the plot
#     plt.figure(figsize=(12, 8))
#     sns.lineplot(
#         data=df_melted,
#         x='boosting_round',
#         y=dress(performance_type),
        
#         estimator = None,       # individual lines instead of error bars

#         hue='outlier_proportion',
#         palette="viridis",

#         style='Dataset',
#         # style_order=["Test", "Train (contaminated)"],
#         legend="full"
#     )
    
#     # Add labels and legend
#     plt.title(f'{dress(performance_type)} vs Boosting Round')
#     plt.xlabel('Boosting Round')
#     plt.ylabel(dress(performance_type))
#     plt.legend(
#         title='Outlier Proportion / Dataset',
#         loc='center left',
#         bbox_to_anchor=(1, 0.5)
#     )
#     plt.grid(True)
    
#     # Return the figure
#     return plt.gcf()