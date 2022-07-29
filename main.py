from dash import Dash, dcc, Input, Output, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from model import create_model
import dash
import numpy as np

DATA_PATH = 'data/iris.csv'

df = pd.read_csv(DATA_PATH)
model, acc_train, acc_test, conf_train, conf_test = create_model(df)

app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
title = dcc.Markdown(children="Test Example")

# Charts
title_distr = dcc.Markdown(children='')
graph_distr = dcc.Graph(figure={})
dropdown_distr = dcc.Dropdown(options=df.columns.values[:-1],
                              value='sepal.length',
                              clearable=False)


@app.callback(
    Output(graph_distr, component_property='figure'),
    Output(title_distr, component_property='children'),
    Input(dropdown_distr, component_property='value')
)
def update_dist_graph(col_name):
    if col_name == 'sepal.width':
        fig = px.histogram(data_frame=df, x='sepal.width', color='variety', opacity=.7)
    elif col_name == 'sepal.length':
        fig = px.histogram(data_frame=df, x='sepal.length', color='variety', opacity=.7)
    elif col_name == 'petal.width':
        fig = px.histogram(data_frame=df, x='petal.width', color='variety', opacity=.7)
    elif col_name == 'petal.length':
        fig = px.histogram(data_frame=df, x='petal.length', color='variety', opacity=.7)
    return fig, col_name


title_sep_pet = dcc.Markdown(children='')
graph_sep_pet = dcc.Graph(figure={})
dropdown_sep_pet = dcc.Dropdown(options=['Sepal Plot', 'Petal Plot'],
                                value='Sepal Plot',
                                clearable=False)


@app.callback(
    Output(graph_sep_pet, component_property='figure'),
    Output(title_sep_pet, component_property='children'),
    Input(dropdown_sep_pet, component_property='value')
)
def update_sep_pet_graph(user_input):
    if user_input == 'Sepal Plot':
        fig = px.scatter(data_frame=df, x='sepal.length', y='sepal.width', color='variety')
    elif user_input == 'Petal Plot':
        fig = px.scatter(data_frame=df, x='petal.length', y='petal.width', color='variety')
    return fig, user_input


fig_heatmap_train = px.imshow(conf_train, text_auto=True)
fig_heatmap_test = px.imshow(conf_test, text_auto=True)


preds_label = dbc.Label(children='')
button = dbc.Button("Get Prediction", color='primary', active=True, n_clicks=0, id='button')
inp1 = dbc.Input(value="", id='inp_sep_len')
inp2 = dbc.Input(value="", id='inp_sep_wid')
inp3 = dbc.Input(value="", id='inp_pet_len')
inp4 = dbc.Input(value="", id='inp_pet_wid')
@app.callback(
    Output(preds_label, component_property='children'),
    Input(button, 'n_clicks'),
    Input(inp1, 'value'),
    Input(inp2, 'value'),
    Input(inp3, 'value'),
    Input(inp4, 'value')
)
def button_clicked(button, inp1, inp2, inp3, inp4):
    label = ''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button' in changed_id:
        try:
            x1 = float(inp1)
            x2 = float(inp2)
            x3 = float(inp3)
            x4 = float(inp4)
        except Exception as e:
            return "Ensure all values were inserted correctly"
        try:
            df = pd.DataFrame({'sepal.length': x1, 'sepal.width': x2, 'petal.length': x3, 'petal.width': x4}, index=[0])
            label = f"The prediction is {model.predict(df)[0]}," \
                    f" probability is {np.round(100*np.max(model.predict_proba(df)),2)}%"
        except Exception as e:
            return "Something went wrong with the model"

    return label

tab1_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label(children='Sepal Length'),
                inp1,
            ], width=5),
            dbc.Col([
                dbc.Label(children='Sepal Width'),
                inp2,
            ], width=5)
        ], justify='center'),
        dbc.Row([
            dbc.Col([
                dbc.Label(children='Petal Length'),
                inp3,
            ], width=5),
            dbc.Col([
                dbc.Label(children='Petal Width'),
                inp4,
            ], width=5)
        ], justify='center'),
        dbc.Label(children=''),
        dbc.Row([
            button
        ], justify='center'),
        preds_label]
    )
)
tab2_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                title_distr,
            ]),
        ], justify='center'),
        dbc.Row([
            dbc.Col([
                graph_distr,
            ]),
        ], justify='center'),
        dbc.Row([
            dbc.Col([
                dropdown_distr,
            ], width=5),
        ], justify='center'),
        dbc.Row([
            dbc.Col([
                title_sep_pet,
            ]),
        ], justify='center'),
        dbc.Row([
            dbc.Col([
                graph_sep_pet,
            ]),
        ], justify='center'),
        dbc.Row([
            dbc.Col([
                dropdown_sep_pet,
            ], width=5),
        ], justify='center'),
        dbc.Label(children=f'The accuracy on the train set is {np.round(100*acc_train, 2)}%, '
                           f'the accuracy on the test set is {np.round(100*acc_test, 2)}%'),
        dbc.Row([
            dbc.Col([
                dbc.Label(children='Train Set Confusion Matrix'),
                dcc.Graph(figure=fig_heatmap_train),
            ], width=5),
            dbc.Col([
                dbc.Label(children='Test Set Confusion Matrix'),
                dcc.Graph(figure=fig_heatmap_test),
            ], width=5)
        ], justify='center'),
    ]
    ),
    className='mt-3'
)
tabs = dbc.Tabs([dbc.Tab(tab1_content, label='Model Prediction'), dbc.Tab(tab2_content, label='Dashboards')])


app.layout = dbc.Container([title, tabs])


if __name__ == '__main__':
    app.run(port=1234, debug=True)