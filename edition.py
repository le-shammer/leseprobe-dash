import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc
from functools import lru_cache

na_text = 'n/a'

# Read CSV file
try:
    df = pd.read_csv("leseprobe.csv", sep=";", decimal=".")
    df.fillna(na_text, inplace=True)
except FileNotFoundError:
    df = pd.read_csv("example_data.csv")

# Function to transform data to long format
def transform_data(df):
    transformed_data = {}
    for letter in ["a", "b", "c", "i", "d"]:
        data = []
        for _, row in df.iterrows():
            for col in df.columns:
                if col.startswith('hs') and col != 'hs'+letter:
                    other_letter = col.replace('hs', '')
                    main_text = row['hs{}'.format(letter)]
                    formatted_letter = other_letter if other_letter in ['i', 'd'] else other_letter.upper()
                    levd_column = [
                        c for c in df.columns if 'levd' in c and 'hs'+letter in c and 'hs'+other_letter in c][0]
                    additional_text = '{}: {}'.format(
                        formatted_letter, row['hs{}'.format(other_letter.lower())])
                    distance = row[levd_column]
                    wmd_column = [
                        c for c in df.columns if 'wmd' in c and 'hs'+letter in c and 'hs'+other_letter in c][0]
                    wmd = row[wmd_column]
                    data.append(
                        [row[0], main_text, additional_text, distance, wmd])
        col_name_reference = 'Referenz-Handschrift Hs. ' + letter.upper()
        transformed_data[letter] = pd.DataFrame(data, columns=[
                                                'Vers', col_name_reference, 'Vergleichshandschriften', 'Distance', "wmd"])
        transformed_data[letter]['Vergleichshandschriften mit Distanzen'] = transformed_data[letter]['Vergleichshandschriften'] + \
            ' (levd: ' + transformed_data[letter]['Distance'].apply(lambda x: '{:.2f}'.format(
                x)) + '; wmd: ' + transformed_data[letter]['wmd'].apply(lambda x: '{:.2f}'.format(x)) + ')'
    return transformed_data

# Function to filter data


def filter_data(min_threshold, max_threshold, min_wmd_threshold, max_wmd_threshold, df, hide_extra_verses, show_distances, combination_logic = 'or'):
    filtered_df = df.copy()
    if show_distances:
        compare_columns = 'Vergleichshandschriften mit Distanzen'
    else:
        compare_columns = 'Vergleichshandschriften'
    col_name_reference = [c for c in df.columns if 'Referenz' in c][0]
    if hide_extra_verses:
        filtered_df = filtered_df[filtered_df[col_name_reference] != na_text]
    if combination_logic == 'and':
        filtered_df.loc[(filtered_df['Distance'] < min_threshold) | (filtered_df['Distance'] > max_threshold) | (
            filtered_df['wmd'] < min_wmd_threshold) | (filtered_df['wmd'] > max_wmd_threshold), compare_columns] = ''
    elif combination_logic == 'or':
        filtered_df.loc[((filtered_df['Distance'] < min_threshold) | (filtered_df['Distance'] > max_threshold)) &
                        ((filtered_df['wmd'] < min_wmd_threshold) | (filtered_df['wmd'] > max_wmd_threshold)), compare_columns] = ''

    grouped_df = filtered_df.groupby(['Vers', col_name_reference])[
        compare_columns].apply(lambda x: '<br>'.join(x)).reset_index()
    return grouped_df


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Transform data to long format
transformed_data = transform_data(df)

# App layout
app.layout = html.Div([
    dbc.NavbarSimple(
        children=[],
        brand="Leseansicht",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Button("Filtermenü", id="toggle-sidebar",
                           className="mb-3"),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            html.H3("Filterkriterien"),
                            html.Label("Referenz-Handschrit wählen"),
                            dcc.Dropdown(
                                id='main-text-dropdown',
                                options=[
                                    {'label': 'Hs. A', 'value': 'a'},
                                    {'label': 'Hs. B', 'value': 'b'},
                                    {'label': 'Hs. C', 'value': 'c'},
                                    {'label': 'Hs. D', 'value': 'd'},
                                    {'label': 'Hs. I', 'value': 'i'}
                                ],
                                value='a'
                            ),
                            html.Label("Levenshtein Distance Range:"),
                            dcc.RangeSlider(
                                id='threshold-slider',
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=[0.0, 1.0],
                                marks={i/20: str(i/20) for i in range(21)}
                            ),
                            html.Label("Word Movers Distance Range:"),
                            dcc.RangeSlider(
                                id='wmd-slider',
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=[0.0, 1.0],
                                marks={i/20: str(i/20) for i in range(21)}
                            ),
                            dcc.Checklist(
                                id='hide-extra-verses',
                                options=[
                                    {'label': '\tZusatzverse ausblenden', 'value': 'hide'}],
                                value=[]
                            ),
                            dcc.Checklist(
                                id='show-distances',
                                options=[
                                    {'label': '\tDistanzen einblenden', 'value': 'show'}],
                                value=[]
                            ),
                            html.Label("Kombinations-Logik"),
                            dcc.Dropdown(
                                id='combination-logic-dropdown',
                                options=[
                                    {'label': 'und', 'value': 'and'},
                                    {'label': 'oder', 'value': 'or'}
                                ],
                                value='and'
                            )
                        ])
                    ),
                    id="sidebar-collapse",
                )
            ], width=3, style={"position": "sticky", "top": 0, "height": "100vh", "overflowY": "auto"}),
            dbc.Col([
                html.Div(id='filtered-texts')
            ], width=9)
        ])
    ], className='container')
])

# Callback to toggle sidebar


@app.callback(
    Output("sidebar-collapse", "is_open"),
    [Input("toggle-sidebar", "n_clicks")],
    [dash.dependencies.State("sidebar-collapse", "is_open")],
)
def toggle_sidebar(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback to update filtered texts


@app.callback(
    Output('filtered-texts', 'children'),
    [Input('main-text-dropdown', 'value'),
     Input('threshold-slider', 'value'),
     Input('wmd-slider', 'value'),
     Input('hide-extra-verses', 'value'),
     Input('show-distances', 'value'),
     Input('combination-logic-dropdown', 'value')]
)
def update_filtered_texts(main_text, levd_threshold_range, wmd_threshold_range, hide_extra_verses, show_distances, combination_logic):
    min_threshold, max_threshold = levd_threshold_range
    min_wmd_threshold, max_wmd_threshold = wmd_threshold_range
    filtered_df = filter_data(min_threshold, max_threshold, min_wmd_threshold, max_wmd_threshold,
                              transformed_data[main_text], 'hide' in hide_extra_verses, 'show' in show_distances, combination_logic)
    if not filtered_df.empty:
        table_rows = []
        for i in range(len(filtered_df)):
            table_rows.append(html.Tr([html.Td(str(filtered_df.iloc[i][col]).replace(
                '<br>', '\n')) for col in filtered_df.columns], style={'background-color': 'lightgrey' if i % 2 == 1 else 'white'}))

        return html.Div([
            html.H3("Leseansicht nach Filterkriterien:"),
            html.Table([
                html.Tr([html.Th(col, style={'width': '5%' if col == 'Vers' else '35%' if 'Referenz-Handschrift' in col else '60%', 'overflow': 'hidden', 'text-overflow': 'ellipsis'}) for col in filtered_df.columns]),
                *table_rows
            ], className='table', style={'whiteSpace': 'pre-wrap', 'table-layout': 'fixed', 'width': '100%'})

        ])
    else:
        return "No filtered texts found."


if __name__ == '__main__':
    app.run_server(debug=True, port="10000", host="0.0.0.0")
