import pandas as pd
from src.feature_engineering import transform_data



if __name__ == "__main__":
    # Static paths relative to the repository
    raw_path = "../data/raw/adr6921_Suppl_Excel_v2.csv"
    output_path='data/processed/'

    # Read in raw data
    df = pd.read_csv(raw_path, encoding_errors='replace')

    # Transform data and obtain final dataframe with counts of factories by country and attribute category
    df = transform_data(df, output_path)