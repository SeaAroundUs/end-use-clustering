import pandas as pd
from src.feature_engineering import transform_data
from src.dimension_reduction import dimension_reduction
from src.top_categories_per_type import plot_top_categories
from src.gdp_lat_cleaning import gdp_lat_cleaning
from src.hierarchical_clustering import plot_heatmap, plot_dendrogram

if __name__ == "__main__":
    # Static paths relative to the repository
    raw_path = "../data/raw/adr6921_Suppl_Excel_v2.csv"
    gdp_path = "../data/raw/worldbank/gdp_per_capita_world_bank.csv"
    mean_lat_path = "../data/raw/geo_mean_location.csv"
    output_path='../data/processed/'
    graph_output_path = '../results/'

    # Read in raw data
    df = pd.read_csv(raw_path, encoding_errors='replace')
    gdp_per_capita = pd.read_csv(gdp_path)
    mean_latitude = pd.read_csv(mean_lat_path)

    # Transform data and obtain final dataframe with counts of factories by country and attribute category
    df = transform_data(df, True, output_path)

    # Set 'country' as index
    X = df.set_index('country')
    # obtain features for clustering
    X = X.drop(columns=['total_factories'])

    # Perform dimensionality reduction using NMF
    W, H = dimension_reduction(X)   

    # Obtain list of countries in W
    countries = W.index.tolist()

    # Create with top categories contributing to each type
    top_category = plot_top_categories(H, 30, True, graph_output_path)

    # clean gpd and mean latitude dataframes for merging with W
    gdp_lat_data = gdp_lat_cleaning(gdp_per_capita, mean_latitude, countries, '2024')

    # Drop countries do not have factory data
    cleaned_W = W.dropna()

    # Merge W with cleaned GDP and latitude dataframes
    final_df = cleaned_W.join(gdp_lat_data)

    # Hierarchical clustering of final dataframe and save visualization of heatmaps and dendrograms    
    plot_heatmap(final_df, save_fig= True, output_path=graph_output_path)
    plot_dendrogram(final_df, save_fig=True, output_path=graph_output_path)