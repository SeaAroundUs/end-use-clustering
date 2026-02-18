import pandas as pd

def transform_data(df: pd.DataFrame, save_df=False, output_path=None) -> pd.DataFrame:
    """
    Transforms the input DataFrame by handling missing values and removing duplicates.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be transformed.
    save_df (bool): Whether to save the processed DataFrame to a CSV file. Default is False.
    output_path (str): The path where the processed data should be saved.

    Returns:
    pd.DataFrame: The transformed DataFrame.
    """
    # Aggregate data by country through pivot tables of different attributes
    # Each pivot table counts the number of factories per country for each attribute category

    # Total number of unique factories per country
    factory_pivot = df.pivot_table(
        index='country', values='factory_id', aggfunc='nunique')
    # Number of factories per country per material type
    material_pivot = df.pivot_table(index='country', columns=[
                                    'material_type'], values='factory_id', aggfunc='nunique', fill_value=0)
    # Number of factories per country per common name
    common_name_pivot = df.pivot_table(index='country', columns=[
                                    'comm_name'], values='factory_id', aggfunc='nunique', fill_value=0)
    # Number of factories per country per scientific name
    sci_name_pivot = df.pivot_table(index='country', columns=[
                                    'sci_name'], values='factory_id', aggfunc='nunique', fill_value=0)
    # Number of factories per country per functional group
    func_group_pivot = df.pivot_table(index='country', columns=[
                                    'func_group'], values='factory_id', aggfunc='nunique', fill_value=0)
    
    # Drop squid column from comm_name_pivot because that is present in func_group_pivot already and is just duplicated
    common_name_pivot = common_name_pivot.drop(columns=['Squid'])

    # Rename factory_id column in factory_pivot to total_factories for clarity
    factory_pivot = factory_pivot.rename(columns={'factory_id': 'total_factories'})

    # Obtain list of unique countries in dataset and set it as dataframe index
    countries = df['country'].unique()
    countries_df = pd.DataFrame(index=countries)
    countries_df.index.name = 'country'

    # Join all pivot tables into a final dataframe indexed by country
    final_df = countries_df.join(
        [factory_pivot, material_pivot, common_name_pivot, sci_name_pivot, func_group_pivot])

    # reset index to make country a column again
    final_df = final_df.reset_index()

    # Fill Nan values with 0
    final_df = final_df.fillna(0)

    # Write final dataframe to csv
    if save_df == True:
        final_df.to_csv(output_path + 'factory_counts_by_country.csv', index=False)

    # Conver columns with counts of factories to percentages of total factories in that country
    for col in final_df.columns[2:]:
        final_df[col] = round(final_df[col] / final_df['total_factories'], 2)

    return final_df