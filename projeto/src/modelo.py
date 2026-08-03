"""
Modelo preditivo de risco de defasagem para a Passos Mágicos.
Idéia: usar os indicadores do ano t para prever se o aluno estará em
defasagem (Defasagem < 0) no ano t+1.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
)

from etl import construir_long, INDICADORES

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

NUM_FEATURES = ['INDE', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'Idade', 'Tempo_PM']
CAT_FEATURES = ['Genero', 'Pedra']


def montar_pares_temporais(df_long):
    """
    Para cada aluno presente em ano t e ano t+1, constrói uma linha com:
      - indicadores no ano t (features)
      - rótulo: Em_defasagem no ano t+1
    """
    df = df_long.copy()
    df = df.sort_values(['RA', 'Ano'])

    base = df[['RA', 'Ano'] + NUM_FEATURES + CAT_FEATURES].copy()
    alvo = df[['RA', 'Ano', 'Em_defasagem']].copy()
    alvo['Ano_origem'] = alvo['Ano'] - 1
    alvo = alvo.rename(columns={'Em_defasagem': 'alvo'})

    pares = base.merge(
        alvo[['RA', 'Ano_origem', 'alvo']],
        left_on=['RA', 'Ano'],
        right_on=['RA', 'Ano_origem'],
        how='inner'
    )
    pares = pares.drop(columns=['Ano_origem'])
    pares = pares.dropna(subset=['alvo'])
    pares['alvo'] = pares['alvo'].astype(int)
    return pares


def montar_pipeline(modelo):
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
        ]), NUM_FEATURES),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore')),
        ]), CAT_FEATURES),
    ])
    return Pipeline([('pre', pre), ('clf', modelo)])


def treinar_e_avaliar(df_pares, random_state=42):
    X = df_pares[NUM_FEATURES + CAT_FEATURES]
    y = df_pares['alvo']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    candidatos = {
        'logistica': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight='balanced'
        ),
        'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
    }

    resultados = {}
    melhor = None
    melhor_auc = -1

    for nome, est in candidatos.items():
        pipe = montar_pipeline(est)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = pipe.predict(X_te)
        auc = roc_auc_score(y_te, proba)
        resultados[nome] = {
            'auc': auc,
            'classification_report': classification_report(y_te, pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_te, pred).tolist(),
            'pipe': pipe,
        }
        if auc > melhor_auc:
            melhor_auc = auc
            melhor = nome

    return resultados, melhor, (X_tr, X_te, y_tr, y_te)


def salvar_modelo(pipe, nome='modelo_risco.joblib'):
    os.makedirs(MODEL_DIR, exist_ok=True)
    caminho = os.path.join(MODEL_DIR, nome)
    joblib.dump(pipe, caminho)
    return caminho


def carregar_modelo(nome='modelo_risco.joblib'):
    caminho = os.path.join(MODEL_DIR, nome)
    return joblib.load(caminho)


if __name__ == '__main__':
    df_long = construir_long()
    pares = montar_pares_temporais(df_long)
    print('linhas treinaveis:', len(pares))
    print('proporcao positiva:', pares['alvo'].mean().round(3))

    resultados, melhor, _ = treinar_e_avaliar(pares)
    for nome, r in resultados.items():
        print(f'{nome}: AUC={r["auc"]:.4f}')
    print('melhor:', melhor)

    pipe_final = resultados[melhor]['pipe']
    caminho = salvar_modelo(pipe_final)
    print('salvo em:', caminho)
