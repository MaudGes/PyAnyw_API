import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import joblib
import shap
import numpy as np

# Charger le pipeline et les données
pipeline = joblib.load('/home/MaudGes/mysite/pipeline_clients_traintest_4.joblib')
# Charger DataFrame des clients
df_clients = pd.read_csv('credit_files/cust_dash.csv') 

# Définition de la liste des features
FEATURE_NAMES = [
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_INCOME_TYPE_Working",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE_Cash loans",
    "REGION_RATING_CLIENT",
    "FLAG_DOCUMENT_3"
]

# Pré-calculer la prédiction et la probabilité pour chaque client
df_clients['probability'] = pipeline.predict_proba(df_clients[FEATURE_NAMES])[:, 1]
df_clients['score'] = (df_clients['probability'] >= 0.15).astype(int)

# Calculer la feature importance globale
# Créer un explainer global
explainer = shap.Explainer(pipeline['model'])  # Attention : pour XGBClassifier, vous pourriez utiliser TreeExplainer
# Calculer les valeurs SHAP pour un échantillon ou pour l’ensemble
shap_values_global = explainer(df_clients[FEATURE_NAMES])

# Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout du dashboard
app.layout = dbc.Container([
    html.H1("Dashboard de prédiction du remboursement de crédit"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Sélectionner un client"),
            dcc.Dropdown(
                id='client-dropdown',
                options=[{'label': f"Client {i}", 'value': i} for i in df_clients.index],
                value=df_clients.index[0]
            )
        ], width=4),
    ], className="my-3"),
    
    dbc.Row([
        dbc.Col([
            html.H3("Score et Probabilité"),
            html.Div(id='score-output'),
            html.Div(id='probability-output'),
        ], width=4),
        dbc.Col([
            html.H3("Contributions des Features"),
            dcc.Graph(id='shap-graph')
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H3("Comparaison avec d'autres clients"),
            dcc.Dropdown(
                id='filter-dropdown',
                options=[{'label': feature, 'value': feature} for feature in FEATURE_NAMES],
                value=FEATURE_NAMES[0]  # Par défaut, on choisit la première feature
            ),
            dcc.Graph(id='comparative-graph')
        ])
    ])
], fluid=True)

# Callback pour mettre à jour le score et la probabilité du client sélectionné
@app.callback(
    Output('score-output', 'children'),
    Output('probability-output', 'children'),
    Input('client-dropdown', 'value')
)
def update_score(client_index):
    # Récupérer le client sélectionné
    client_data = df_clients.loc[client_index]
    prob = client_data['probability']
    score = client_data['score']
    # Exemple d'interprétation simplifiée
    interpretation = ("Client à risque élevé" if score == 1 else "Client à faible risque")
    return (
        f"Score: {score} - {interpretation}",
        f"Probabilité de non-remboursement: {prob:.2%}"
    )

# Callback pour mettre à jour le graphique SHAP pour le client sélectionné
@app.callback(
    Output('shap-graph', 'figure'),
    Input('client-dropdown', 'value')
)
def update_shap_graph(client_index):
    # Calculer les valeurs SHAP locales pour le client
    client_data = df_clients.loc[[client_index]][FEATURE_NAMES]
    shap_values_local = explainer(client_data)
    # Créer un graphique à barres pour les contributions SHAP
    shap_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Contribution': shap_values_local.values[0]
    }).sort_values(by='Contribution', key=abs, ascending=False)
    
    fig = px.bar(shap_df, x='Feature', y='Contribution',
                 title="Contribution des features pour ce client")
    return fig

# Callback pour mettre à jour le graphique comparatif en fonction du filtre choisi
@app.callback(
    Output('comparative-graph', 'figure'),
    Input('filter-dropdown', 'value'),
    Input('client-dropdown', 'value')
)
def update_comparative_graph(selected_feature, client_index):
    # Filtrer (ou simplement récupérer) la distribution de la feature sélectionnée
    fig = px.histogram(
        df_clients,
        x=selected_feature,
        nbins=50,
        title=f"Distribution de {selected_feature} chez tous les clients"
    )
    # Ajouter la valeur du client sélectionné
    client_value = df_clients.loc[client_index, selected_feature]
    fig.add_vline(
        x=client_value,
        line_dash="dash",
        line_color="red",
        annotation_text="Valeur du client",
        annotation_position="top right"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)