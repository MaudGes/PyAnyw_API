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
import plotly.graph_objects as go

# ----------------------------
# Partie Flask
# ----------------------------
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
df_clients = pd.read_csv('/home/MaudGes/mysite/credit_files/cust_dash.csv')

# Calculer la probabilité et le score pour chaque client
df_clients['probability'] = pipeline.predict_proba(df_clients[FEATURE_NAMES])[:, 1]
df_clients['score'] = (df_clients['probability'] >= OPTIMAL_THRESHOLD).astype(int)

# Créer un explainer pour SHAP (pour XGBClassifier, on peut aussi utiliser TreeExplainer)
explainer = shap.Explainer(pipeline['model'])
# Calculer (optionnellement) les valeurs SHAP globales pour toutes les lignes
shap_values_global = explainer(df_clients[FEATURE_NAMES])
# Calcul de l'importance globale : moyenne des valeurs absolues de SHAP pour chaque feature
global_importance = np.abs(shap_values_global.values).mean(axis=0)
global_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Global Importance': global_importance
})

# Création de l'application Dash en utilisant le serveur Flask
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname='/dashboard/',
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Layout du dashboard Dash avec attributs d’accessibilité
dash_app.layout = dbc.Container([
    # Navbar de navigation avec aria‑label
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink("Prédiction", href="/", **{"aria-label": "Retour à la page de prédiction"})
            ),
        ],
        brand="Dashboard Crédit",
        brand_href="/dashboard/",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Ligne 1 : Sélection du client avec un label explicite
    dbc.Row([
        dbc.Col([
            html.Label("Sélectionner un client", id="client-dropdown-label"),
            dcc.Dropdown(
                id='client-dropdown',
                options=[{'label': f"Client {i}", 'value': i} for i in df_clients.index],
                value=df_clients.index[0]
            )
        ], width=4),
    ], className="my-3"),
    
    # Ligne 2 : Score, probabilité et graphique SHAP local
    dbc.Row([
        dbc.Col([
            html.H3("Score et Probabilité", id="score-prob-title"),
            html.Div(id='score-output'),
            html.Div(id='probability-output'),
        ], width=4),
        dbc.Col([
            html.H3("Contributions des variables pour le client", id="shap-local-title"),
            dcc.Graph(
                id='shap-graph',
                **{"aria-label": "Graphique SHAP local montrant la contribution de chaque feature pour le client."}
            )
        ], width=8)
    ], className="my-3"),
    
    # Ligne 3 : Jauge et comparaison Local vs Global
    dbc.Row([
        dbc.Col([
            html.H3("Indicateur : Écart par rapport au seuil", id="gauge-title"),
            dcc.Graph(
                id='gauge-indicator',
                **{"aria-label": "Graphique de jauge indiquant la probabilité de non-remboursement par rapport au seuil."}
            )
        ], width=6),
        dbc.Col([
            html.H3("Comparaison des variables : Locale vs Globales", id="global-local-title"),
            dcc.Graph(
                id='global-local-graph',
                **{"aria-label": "Graphique comparatif montrant la contribution locale par rapport à l'importance globale des features."}
            )
        ], width=6)
    ], className="my-3"),
    
    # Ligne 4 : Analyse bivariée entre deux features avec des labels explicites
    dbc.Row([
        dbc.Col([
            html.H3("Analyse bivariée", id="bivariate-title"),
            html.Label("Sélectionner la feature X", id="feature-x-label"),
            dcc.Dropdown(
                id='feature-x-dropdown',
                options=[{'label': feature, 'value': feature} for feature in FEATURE_NAMES],
                value=FEATURE_NAMES[0]
            )
        ], width=6),
        dbc.Col([
            html.Label("Sélectionner la feature Y", id="feature-y-label"),
            dcc.Dropdown(
                id='feature-y-dropdown',
                options=[{'label': feature, 'value': feature} for feature in FEATURE_NAMES],
                value=FEATURE_NAMES[1]
            )
        ], width=6)
    ], className="my-3"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='bivariate-graph',
                **{"aria-label": "Graphique d'analyse bivariée affichant la relation entre les deux features sélectionnées."}
            )
        ])
    ], className="my-3"),
    
    # Ligne 5 : Comparaison distribution d'une feature avec un label explicite
    dbc.Row([
        dbc.Col([
            html.H3("Comparaison avec d'autres clients", id="comparative-title"),
            html.Label("Sélectionner une feature pour comparer sa distribution parmi tous les clients", id="filter-dropdown-label"),
            dcc.Dropdown(
                id='filter-dropdown',
                options=[{'label': feature, 'value': feature} for feature in FEATURE_NAMES],
                value=FEATURE_NAMES[0]
            ),
            dcc.Graph(
                id='comparative-graph',
                **{"aria-label": "Graphique de distribution comparant la valeur d'une feature chez le client sélectionné à celle de l'ensemble des clients."}
            )
        ])
    ], className="my-3")
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

# Callback pour mettre à jour le graphique SHAP local pour le client sélectionné
@dash_app.callback(
    Output('shap-graph', 'figure'),
    Input('client-dropdown', 'value')
)
def update_shap_graph(client_index):
    client_data = df_clients.loc[[client_index]][FEATURE_NAMES]
    shap_values_local = explainer(client_data)
    shap_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Local Contribution': shap_values_local.values[0]
    }).sort_values(by='Local Contribution', key=abs, ascending=False)
    
    fig = px.bar(shap_df, x='Feature', y='Local Contribution',
                 title="Contribution des variables pour ce client")
    fig.update_layout(
        title={'text': "Les barres indiquent l'impact de chaque feature sur la prédiction.", 'x':0.5}
    )
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

# Callback pour mettre à jour la jauge indiquant l'écart par rapport au seuil
@dash_app.callback(
    Output('gauge-indicator', 'figure'),
    Input('client-dropdown', 'value')
)
def update_gauge(client_index):
    client_prob = df_clients.loc[client_index, 'probability']
    color = "green" if client_prob < 0.15 else "red"
    
    fig = go.Figure()
    
    # Barre de fond (grise)
    fig.add_shape(
        type="rect",
        x0=0, x1=1, y0=0.4, y1=0.6,
        fillcolor="lightgrey",
        line=dict(width=0),
        name="Échelle complète"
    )
    
    # Barre colorée représentant la probabilité du client
    fig.add_shape(
        type="rect",
        x0=0, x1=client_prob, y0=0.4, y1=0.6,
        fillcolor=color,
        line=dict(width=0),
        name="Valeur du client"
    )
    
    # Ligne verticale indiquant la position exacte du client
    fig.add_shape(
        type="line",
        x0=client_prob, x1=client_prob,
        y0=0.35, y1=0.65,
        line=dict(color="black", width=4)
    )
    
    # Ligne verticale pour le seuil de 15%
    fig.add_shape(
        type="line",
        x0=0.15, x1=0.15,
        y0=0.3, y1=0.7,
        line=dict(color="blue", width=3, dash="dash"),
        name="Seuil 15%"
    )

    # Annotation pour la valeur du client
    fig.add_annotation(
        x=client_prob, y=0.75,
        text=f"{client_prob*100:.1f}%",
        showarrow=False,
        font=dict(size=14, color="black")
    )

    # Annotation pour le seuil à 15%
    fig.add_annotation(
        x=0.15, y=0.3,  # Déplace l'annotation en dessous
        text="Seuil 15%",
        showarrow=False,
        font=dict(size=14, color="black")
    )

    fig.update_layout(
        title=dict(
            text="Indicateur : Probabilité de non-remboursement",
            y=0.85
        ),
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            tickmode='linear',
            dtick=0.1,
            tickformat=".0%"
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=False,
            showticklabels=False
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=150
    )

    return fig

# Callback pour comparer la contribution locale du client avec l'importance globale
@dash_app.callback(
    Output('global-local-graph', 'figure'),
    Input('client-dropdown', 'value')
)
def update_global_local_graph(client_index):
    client_data = df_clients.loc[[client_index]][FEATURE_NAMES]
    shap_values_local = explainer(client_data)
    local_contrib = np.abs(shap_values_local.values[0])
    compare_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Local Contribution': local_contrib,
        'Global Importance': global_importance
    })
    fig = px.bar(compare_df, x='Feature', y=['Local Contribution', 'Global Importance'],
                 barmode='group', title="Comparaison : Contribution Locale vs Importance Globale")
    fig.update_layout(
        title={'text': "Graphique comparatif de la contribution locale (pour ce client)<br>et de l'importance globale des features", 'x':0.5}
    )
    return fig

# Callback pour réaliser une analyse bivariée entre deux features sélectionnées
@dash_app.callback(
    Output('bivariate-graph', 'figure'),
    Input('feature-x-dropdown', 'value'),
    Input('feature-y-dropdown', 'value')
)
def update_bivariate_graph(feature_x, feature_y):
    fig = px.scatter(df_clients, x=feature_x, y=feature_y,
                     title=f"Analyse bivariée : {feature_x} vs {feature_y}",
                     trendline="ols")
    fig.update_layout(
        title={'text': f"Nuage de points et tendance linéaire entre {feature_x} et {feature_y}", 'x':0.5}
    )
    return fig