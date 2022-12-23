import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

df = pd.read_csv('sp500_Ffilled.csv')


def set_date_index(input_df, col_name='date'):
    """Given a pandas df, parse and set date column to index.
        col_name will be removed and set as datetime index.

    Args:
        input_df (pandas dataframe): Original pandas dataframe
        col_name (string): Name of date column

    Returns:
        pandas dataframe: modified and sorted dataframe
    """
    # Copy df to prevent changing original
    modified_df = input_df.copy()

    # Infer datetime from col
    modified_df[col_name] = pd.to_datetime(modified_df[col_name])

    # Sort and set index
    modified_df.sort_values(col_name, inplace=True)
    modified_df.set_index(col_name, inplace=True)

    return modified_df


df = set_date_index(df)


def plot_ts_data(plot_df):
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(x=plot_df.index, y=plot_df.value, ax=ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))


df_bfill = df.resample('1D').mean().ffill()

df_bfill.to_csv("sp500_Ffilled_mean_only.csv")
