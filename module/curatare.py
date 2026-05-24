import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st


@st.cache_data
def incarca_date():
    """Incarca setul de date din CSV si pastreaza coloanele relevante."""
    df = pd.read_csv("Data/Car_Insurance_Claim.csv")
    
    # Selectam doar coloanele necesare
    cols_selectate = [
        'ID', 'AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'INCOME',
        'CREDIT_SCORE', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN',
        'ANNUAL_MILEAGE', 'VEHICLE_TYPE', 'SPEEDING_VIOLATIONS',
        'PAST_ACCIDENTS', 'OUTCOME'
    ]
    
    df = df[cols_selectate].copy()
    return df


def curata_date(df):
    """Returneaza un dataframe curatat complet cu toate transformarile."""
    df_curat = df.copy()
    
    # 1. Completare valori lipsa numerice cu media
    cols_numerice = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'MARRIED', 'CHILDREN']
    for col in cols_numerice:
        if col in df_curat.columns:
            df_curat[col] = df_curat[col].fillna(df_curat[col].mean())
    
    # 2. Completare valori lipsa categorice cu modul (daca exista)
    cols_categorice = ['AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'INCOME', 
                       'VEHICLE_YEAR', 'VEHICLE_TYPE']
    for col in cols_categorice:
        if col in df_curat.columns and df_curat[col].isnull().sum() > 0:
            df_curat[col] = df_curat[col].fillna(df_curat[col].mode()[0])
    
    # 3. Eliminare outlieri prin IQR
    cols_outlieri = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 
                     'PAST_ACCIDENTS']
    for col in cols_outlieri:
        if col in df_curat.columns:
            Q1 = df_curat[col].quantile(0.25)
            Q3 = df_curat[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_curat = df_curat[
                (df_curat[col] >= lower) & (df_curat[col] <= upper)
            ]
    
    # 4. Codificare variabile categorice cu LabelEncoder
    cols_encode = ['AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'INCOME', 
                   'VEHICLE_YEAR', 'VEHICLE_TYPE']
    le = LabelEncoder()
    for col in cols_encode:
        if col in df_curat.columns:
            df_curat[col + '_cod'] = le.fit_transform(df_curat[col].astype(str))
    
    # 5. Scalare variabile numerice cu StandardScaler
    cols_scale = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 
                  'PAST_ACCIDENTS']
    scaler = StandardScaler()
    df_curat[[c + '_scaled' for c in cols_scale]] = scaler.fit_transform(
        df_curat[cols_scale]
    )
    
    return df_curat


def get_outlieri_info(df):
    """Returneaza informatii despre outlieri per coloana."""
    cols_outlieri = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 
                     'PAST_ACCIDENTS']
    outlieri_info = {}
    limite_info = []
    
    for col in cols_outlieri:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_out = ((df[col] < lower) | (df[col] > upper)).sum()
            outlieri_info[col] = int(n_out)
            limite_info.append({
                "Coloană": col,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "Lower bound": round(lower, 2),
                "Upper bound": round(upper, 2),
                "Outlieri găsiți": int(n_out)
            })
    
    return outlieri_info, limite_info