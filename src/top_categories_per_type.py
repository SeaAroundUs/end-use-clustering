import altair as alt
import pandas as pd

def plot_top_categories(H, top_n=30, save_chart= False, output_path='../data/graphs/'):
    """
    Create bar charts for top contributing categories for each type.
    
    Parameters:
    H (pd.DataFrame): dataframe of shape (n_components, n_features) containing the definition of each component in terms of the original features.
    top_n (int): number of top categories to display for each type.
    save_chart (bool): whether to save the generated chart as an HTML file. Default is False.
    output_path (str): path to save the generated charts. Default is 'data/graphs/'.

    """

    # Pivot H to long format for Altair
    H_long = H.reset_index().melt(
        id_vars="index",
        var_name="Category",
        value_name="Weight"
        )
    
    # Identify top 30 categories contributing to each component
    top_categories = (H_long
                    .groupby(['Component'])
                    .apply(lambda x: x.nlargest(30, 'Weight'))
                    .reset_index()
                    .drop(columns=['level_1']))
    
    bar_chart = alt.Chart(top_categories).mark_bar().encode(
        x=alt.X("Weight:Q", title="Contribution"),
        y=alt.Y("Category:N", sort='-x'),
        tooltip=["Component", "Category", "Weight"]
        ).facet(
            row=alt.Row("Component:N", title="NMF Components")
        ).resolve_scale(
            y="independent"
        )
    
    # Save the chart as an HTML file
    if save_chart:
        bar_chart.save(f"{output_path}top_categories_per_type.html")

    return bar_chart