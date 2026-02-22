from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(data, save_fig=False, output_path=None):
    """
    Perform hierarchical clustering and plot heatmap with dendrograms.

    Parameters:
    data (pd.DataFrame): The input data for clustering, indexed by country.
    save_fig (bool): Whether to save the generated heatmap as an image file. Default is False.
    output_path (str): The path to save the generated heatmap.

    Returns:
    None
    """
    # Plot the heatmap on the right side
    heatmap = sns.clustermap(data, method='ward', cmap='coolwarm', figsize=(
    20, 20), cbar_pos=(0, 0.8, .03, .2), row_cluster=True, col_cluster=False)
    
    # Save the plot as an image file
    if save_fig:
        heatmap.savefig(f"{output_path}hierarchical_clustering_heatmap.png", bbox_inches='tight')
    else:
        plt.show()


def plot_dendrogram(data, save_fig=False, output_path=None):
    """
    Perform hierarchical clustering and plot dendrogram.

    Parameters:
    data (pd.DataFrame): The input data for clustering, indexed by country.
    save_fig (bool): Whether to save the generated dendrogram as an image file. Default is False.
    output_path (str): The path to save the generated dendrogram.

    Returns:
    None
    
    """
    # Perform hierarchical clustering using Ward's method
    linkage_matrix = ward(data)

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=data.index, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Countries')
    plt.ylabel('Distance')

    # Save the plot as an image file
    if save_fig:
        plt.savefig(f"{output_path}hierarchical_clustering_dendrogram.png", bbox_inches='tight')
    else:
        plt.show()