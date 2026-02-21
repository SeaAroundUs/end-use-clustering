import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def gdp_lat_cleaning(gdp_df: pd.DataFrame, lat_df: pd.DataFrame, 
                     countries:list, gdp_year = '2024') -> (pd.DataFrame, pd.DataFrame):
    """
    Cleans the GDP per capita and mean latitude dataframes for merging with the NMF results.

    Parameters:
    gdp_df (pd.DataFrame): The raw GDP per capita dataframe.
    lat_df (pd.DataFrame): The raw mean latitude dataframe.
    countries (list): List of countries present in the NMF results for filtering the GDP and latitude dataframes.
    gdp_year (str): The year for which to extract GDP per capita data. Default is 2024.

    Returns:
    clean_gdp (pd.DataFrame): The cleaned GDP per capita dataframe with 'country' and 'gdp_per_capita' columns.
    clean_lat (pd.DataFrame): The cleaned mean latitude dataframe with 'country' and 'mean_latitude' columns.
    """
    # Clean GDP per capita dataframe
    clean_gdp = gdp_df.set_index("Country Name")

    # Rename countries in GDP dataframe to match those in W_df for merging
    clean_gdp = clean_gdp.rename(index={'Gambia, The': 'Gambia',
                                              'Iran, Islamic Rep.': 'Iran',
                                              'Korea, Rep.': 'South Korea',
                                              'Russian Federation': 'Russia',
                                              'Turkiye': 'Turkey',
                                              'United States': 'United States of America',
                                              'Viet Nam': 'Vietnam',
                                              'Yemen, Rep.': 'Yemen'})
    
    # Extract GDP per capita for the specified year and rename column
    clean_gdp = clean_gdp[[gdp_year]].rename(columns={gdp_year: 'gdp_per_capita'})

    # Clean mean latitude dataframe
    clean_lat = lat_df.set_index('geo_name')

    # Rename countries in latitude dataframe to match those in W_df for merging
    clean_lat = clean_lat.rename(index={'Faeroe Isl. (Denmark)': 'Faroe Islands',
                                  'Korea (South)': 'South Korea',
                                  'Russian Federation': 'Russia',
                                  'USA': 'United States of America',
                                  'Viet Nam': 'Vietnam'},
                           columns={'lat': 'mean_lat'})
    
    # Filter both dataframes to only include countries present in W_df
    countries_df = pd.DataFrame(index=countries)
    combined_df = countries_df.join([clean_gdp, clean_lat[['mean_lat']]])

    # Scale GDP per capita and mean latitude to be between 0 and 1 for better comparability
    scaler = MinMaxScaler()
    scaled_cols = scaler.fit_transform(combined_df[['gdp_per_capita', 'mean_lat']])

    # Update combined_df with scaled columns
    combined_df[['gdp_per_capita', 'mean_lat']] = scaled_cols

    return combined_df