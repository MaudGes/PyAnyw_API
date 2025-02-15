from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import shap

# Import Dash et ses composants
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# ----------------------------
# Partie Flask
# ----------------------------
# Créez l'application Flask
app = Flask(__name__)

# Charger le pipeline (modèle entraîné)
pipeline = joblib.load('/home/MaudGes/mysite/pipeline_clients_traintest_4.joblib')

# Définition de la liste des features utilisées pour la prédiction
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

# Seuil optimal pour la classification
OPTIMAL_THRESHOLD = 0.15  

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None 

    if request.method == 'POST':
        try:
            # Récupération des données du formulaire et conversion en float/int
            input_data = [
                float(request.form['EXT_SOURCE_3']),
                float(request.form['EXT_SOURCE_2']),
                1 if request.form.get('NAME_EDUCATION_TYPE_Higher education') == 'on' else 0,
                1 if request.form.get('NAME_INCOME_TYPE_Working') == 'on' else 0,
                1 if request.form.get('NAME_EDUCATION_TYPE_Secondary / secondary special') == 'on' else 0,
                int(request.form['CODE_GENDER']),
                1 if request.form.get('NAME_CONTRACT_TYPE_Cash loans') == 'on' else 0,
                int(request.form['REGION_RATING_CLIENT']),
                int(request.form['FLAG_DOCUMENT_3']),
            ]

            # Conversion en DataFrame avec les noms de features
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

            # Vérification du pipeline
            if pipeline is None:
                print("❌ ERROR: Pipeline is None!")
                return render_template('index.html', error="Pipeline is None.")
            else:
                print(f"✅ Pipeline loaded successfully: {type(pipeline)}")

            # Prédiction
            prediction_prob = pipeline.predict_proba(input_df)[0][1]
            prediction = 1 if prediction_prob >= OPTIMAL_THRESHOLD else 0

            print("✅ Prediction Probability:", prediction_prob)
            print("✅ Final Prediction:", prediction)

            probability = round(prediction_prob, 5)

        except Exception as e:
            print(f"❌ Exception Occurred: {e}")
            return render_template('index.html', error=str(e))

    print("🚀 Sending to template:", prediction, probability)
    return render_template('index.html', prediction=prediction, probability=probability)

# ----------------------------
# Partie Dashboard Dash intégrée à Flask
# ----------------------------
# Charger le DataFrame des clients pour le dashboard
df_clients = pd.read_csv('credit_files/cust_dash.csv')

# Calculer la probabilité et le score pour chaque client
df_clients['probability'] = pipeline.predict_proba(df_clients[FEATURE_NAMES])[:, 1]
df_clients['score'] = (df_clients['probability'] >= OPTIMAL_THRESHOLD).astype(int)

# Créer un explainer pour SHAP (pour XGBClassifier, vous pouvez également utiliser TreeExplainer)
explainer = shap.Explainer(pipeline['model'])
# Calculer (optionnellement) les valeurs SHAP globales
shap_values_global = explainer(df_clients[FEATURE_NAMES])

# Créer l'application Dash en utilisant le serveur Flask
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname='/dashboard/',
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Layout du dashboard Dash
dash_app.layout = dbc.Container([
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
            html.H3("Contributions des Features (SHAP)"),
            dcc.Graph(id='shap-graph')
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H3("Comparaison avec d'autres clients"),
            dcc.Dropdown(
                id='filter-dropdown',
                options=[{'label': feature, 'value': feature} for feature in FEATURE_NAMES],
                value=FEATURE_NAMES[0]
            ),
            dcc.Graph(id='comparative-graph')
        ])
    ])
], fluid=True)

# Callback pour mettre à jour le score et la probabilité du client sélectionné
@dash_app.callback(
    Output('score-output', 'children'),
    Output('probability-output', 'children'),
    Input('client-dropdown', 'value')
)
def update_score(client_index):
    client_data = df_clients.loc[client_index]
    prob = client_data['probability']
    score = client_data['score']
    interpretation = "Client à risque élevé" if score == 1 else "Client à faible risque"
    return (
        f"Score: {score} - {interpretation}",
        f"Probabilité de non-remboursement: {prob:.2%}"
    )

# Callback pour mettre à jour le graphique SHAP pour le client sélectionné
@dash_app.callback(
    Output('shap-graph', 'figure'),
    Input('client-dropdown', 'value')
)
def update_shap_graph(client_index):
    client_data = df_clients.loc[[client_index]][FEATURE_NAMES]
    shap_values_local = explainer(client_data)
    shap_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Contribution': shap_values_local.values[0]
    }).sort_values(by='Contribution', key=abs, ascending=False)
    
    fig = px.bar(shap_df, x='Feature', y='Contribution',
                 title="Contribution des features pour ce client")
    return fig

# Callback pour mettre à jour le graphique comparatif en fonction de la feature sélectionnée
@dash_app.callback(
    Output('comparative-graph', 'figure'),
    Input('filter-dropdown', 'value'),
    Input('client-dropdown', 'value')
)
def update_comparative_graph(selected_feature, client_index):
    fig = px.histogram(
        df_clients,
        x=selected_feature,
        nbins=50,
        title=f"Distribution de {selected_feature} chez tous les clients"
    )
    client_value = df_clients.loc[client_index, selected_feature]
    fig.add_vline(
        x=client_value,
        line_dash="dash",
        line_color="red",
        annotation_text="Valeur du client",
        annotation_position="top right"
    )
    return fig