import matplotlib.pyplot as plt
import seaborn as sns

def plot_outlier_density(df, column_name, color='skyblue'):
    """
    Evaluates outlier density using a combined Boxplot, Violin plot, and Stripplot.
    Helps visualize if log transformation will help KMeans clustering.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    column_name : str
        The name of the column to analyze.
    color : str, default='skyblue'
        The base color for the plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribution & Outlier Analysis for {column_name}', fontsize=16, fontweight="bold")

    # 1. Boxplot (Standard outlier detection)
    sns.boxplot(x=df[column_name], ax=axes[0], color=color)
    axes[0].set_title('Boxplot (IQR Method)')

    # 2. Violin Plot (Density estimation)
    sns.violinplot(x=df[column_name], ax=axes[1], color='lightgreen')
    axes[1].set_title('Violin Plot (Density & Spread)')

    # 3. StripPlot (Raw data points)
    sns.stripplot(x=df[column_name], ax=axes[2], color='salmon', alpha=0.3, jitter=True)
    axes[2].set_title('Stripplot (Raw Data Points)')

    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_rfm_distributions(df, columns, colors, hist_title='Distribution of RFM Features', box_title='RFM Features - Outlier Detection'):
    """
    Creates two plots: Histograms and Boxplots for the specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    columns : list of str
        The numeric columns to analyze (e.g., ['Recency', 'Frequency', 'Monetary']).
    colors : list of str
        Colors corresponding to each column for the plots.
    hist_title : str
        Title for the histogram figure.
    box_title : str
        Title for the boxplot figure.
    
    Returns:
    --------
    fig_hist : matplotlib.figure.Figure
        The figure object for the histograms.
    fig_box : matplotlib.figure.Figure
        The figure object for the boxplots.
    """
    n_cols = len(columns)
    
    # 1. Histograms (Distribution)
    fig_hist, axes_hist = plt.subplots(1, n_cols, figsize=(15, 5))
    fig_hist.suptitle(hist_title, fontsize=16, fontweight="bold")

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes_hist[i], color=colors[i], bins=30)
        axes_hist[i].set_title(f'{col} Distribution')
        axes_hist[i].set_xlabel(col)
        axes_hist[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show() # Display immediately

    # 2. Box Plots (Outliers)
    fig_box, axes_box = plt.subplots(1, n_cols, figsize=(15, 5))
    fig_box.suptitle(box_title, fontsize=16, fontweight="bold")

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes_box[i], color=colors[i])
        axes_box[i].set_title(f'{col} Outliers')
        axes_box[i].set_xlabel(col)

    plt.tight_layout()
    plt.show() # Display immediately
    
    return fig_hist, fig_box