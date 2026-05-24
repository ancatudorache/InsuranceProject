import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
import statsmodels.api as sm


def pregateste_date_cluster(df):
    """Pregateste datele pentru clusterizare KMeans."""
    df_cl = df.copy()
    
    # Completam valorile lipsa
    cols_fill = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'MARRIED', 'CHILDREN']
    for col in cols_fill:
        if col in df_cl.columns:
            df_cl[col] = df_cl[col].fillna(df_cl[col].mean())
    
    # Features pentru clusterizare
    features = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 
                'PAST_ACCIDENTS']
    features = [f for f in features if f in df_cl.columns]
    
    df_clean = df_cl[features + ['OUTCOME']].dropna()
    
    # Scalare
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[features])
    
    return df_clean, X, features


def calculeaza_inertii(X, k_max=8):
    """Calculeaza inertiile pentru metoda cotului (Elbow)."""
    inertii = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertii.append(km.inertia_)
    return list(range(2, k_max + 1)), inertii


def aplica_kmeans(X, df_clean, features, k):
    """Aplica KMeans si returneaza dataframe cu clustere."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_result = df_clean.copy()
    df_result['Cluster'] = km.fit_predict(X).astype(str)
    return df_result, km


def pregateste_date_regresie_logistica(df):
    """Pregateste datele pentru regresia logistica."""
    df_rl = df.copy()
    
    # Completare valori lipsa
    cols_fill = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'MARRIED', 'CHILDREN']
    for col in cols_fill:
        if col in df_rl.columns:
            df_rl[col] = df_rl[col].fillna(df_rl[col].mean())
    
    # Codificare categorice (daca nu exista deja coloanele _cod)
    if 'AGE_cod' not in df_rl.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in ['AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'INCOME', 
                    'VEHICLE_YEAR', 'VEHICLE_TYPE']:
            if col in df_rl.columns:
                df_rl[col + '_cod'] = le.fit_transform(df_rl[col].astype(str))
    
    # Features pentru regresie logistica
    features = [
        'CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 
        'PAST_ACCIDENTS', 'MARRIED', 'CHILDREN',
        'AGE_cod', 'GENDER_cod', 'VEHICLE_TYPE_cod'
    ]
    features = [f for f in features if f in df_rl.columns]
    
    df_clean = df_rl[features + ['OUTCOME']].dropna()
    
    # Sample pentru performanta (5000 rânduri)
    n = min(5000, len(df_clean))
    df_sample = df_clean.sample(n, random_state=42)
    
    X = df_sample[features].astype(float)
    y = df_sample['OUTCOME']
    
    # Scalare
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, features


def antreneaza_regresie_logistica(X_train, y_train):
    """Antreneaza modelul de regresie logistica."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def calculeaza_metrici(model, X_test, y_test):
    """Calculeaza metricile de performanta pentru clasificare."""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, zero_division=0) * 100,
        'recall': recall_score(y_test, y_pred, zero_division=0) * 100,
        'f1': f1_score(y_test, y_pred, zero_division=0) * 100,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred
    }


def pregateste_date_ols(df):
    """Pregateste datele pentru regresia OLS."""
    df_ols = df.copy()
    
    # Completare valori lipsa
    cols_fill = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'MARRIED', 'CHILDREN']
    for col in cols_fill:
        if col in df_ols.columns:
            df_ols[col] = df_ols[col].fillna(df_ols[col].mean())
    
    # Features pentru OLS (predictori pentru CREDIT_SCORE ca variabila dependenta)
    features_ols = [
        'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 
        'MARRIED', 'CHILDREN'
    ]
    features_ols = [f for f in features_ols if f in df_ols.columns]
    
    df_clean = df_ols[features_ols + ['CREDIT_SCORE']].dropna()
    
    # Sample pentru performanta
    n = min(5000, len(df_clean))
    df_sample = df_clean.sample(n, random_state=42)
    
    # Preparam X si y
    X_ols = sm.add_constant(df_sample[features_ols].astype(float))
    y_ols = df_sample['CREDIT_SCORE']
    
    return X_ols, y_ols, features_ols, df_sample


def antreneaza_ols(X_ols, y_ols):
    """Antreneaza modelul OLS."""
    return sm.OLS(y_ols, X_ols).fit()