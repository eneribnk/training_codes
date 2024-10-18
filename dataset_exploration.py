import pandas as pd

df = pd.read_csv('merged_1-2116.csv', sep=';')


def print_column_stats(df):
    for column in df.columns:
        col_data = df[column]

        # Calculate statistics
        nan_percentage = col_data.isna().mean() * 100
        min_value = col_data.min()
        max_value = col_data.max()
        mean_value = col_data.mean()
        std_value = col_data.std()

        # Print stats for each column
        print(f"Column: {column}")
        print(f"NaN Percentage: {nan_percentage:.2f}%")
        print(f"Min: {min_value}")
        print(f"Max: {max_value}")
        print(f"Mean: {mean_value}")
        print(f"Std: {std_value}")
        print("-" * 30)


print_column_stats(df)
