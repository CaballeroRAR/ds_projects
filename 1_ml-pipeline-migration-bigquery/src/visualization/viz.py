def plot_outlier_density(df, column_name):
    """
    Evaluates outlier density using a combined Boxplot, Violin plot, and Stripplot.
    Helps visualize if log transformation will help KMeans clustering.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    column_name : str
        The name of the column to analyze.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribution & Outlier Analysis for {column_name}', fontsize=16)

    # 1. Boxplot (Standard outlier detection)
    sns.boxplot(x=df[column_name], ax=axes[0], color='skyblue')
    axes[0].set_title('Boxplot (IQR Method)')

    # 2. Violin Plot (Density estimation)
    sns.violinplot(x=df[column_name], ax=axes[1], color='lightgreen')
    axes[1].set_title('Violin Plot (Density & Spread)')

    # 3. StripPlot (Raw data points)
    sns.stripplot(x=df[column_name], ax=axes[2], color='salmon', alpha=0.5, jitter=True)
    axes[2].set_title('Stripplot (Raw Data Points)')

    plt.tight_layout()
    plt.show()