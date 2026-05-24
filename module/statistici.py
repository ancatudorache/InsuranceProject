import pandas as pd
import numpy as np


def get_distributie_outcome(df):
    """Returneaza distributia clientilor pe outcome (dauna/fara dauna)."""
    out = df['OUTCOME'].value_counts().reset_index()
    out.columns = ['Outcome', 'Nr. clienți']
    out['Outcome'] = out['Outcome'].map({
        0.0: 'Fără daună (0)',
        1.0: 'Cu daună (1)'
    })
    out['Procent (%)'] = (out['Nr. clienți'] / out['Nr. clienți'].sum() * 100).round(1)
    return out.sort_values('Outcome')


def get_distributie_varsta(df):
    """Returneaza distributia pe categorii de varsta."""
    varsta = df.groupby('AGE').agg(
        nr_clienti=('OUTCOME', 'count'),
        rata_daune=('OUTCOME', 'mean')
    ).round(3).reset_index()
    varsta['rata_daune'] = (varsta['rata_daune'] * 100).round(1)
    varsta.columns = ['Categorie vârstă', 'Nr. clienți', 'Rată daune (%)']
    return varsta


def get_distributie_income(df):
    """Returneaza distributia pe categorii de venit."""
    income = df.groupby('INCOME').agg(
        nr_clienti=('OUTCOME', 'count'),
        rata_daune=('OUTCOME', 'mean')
    ).round(3).reset_index()
    income['rata_daune'] = (income['rata_daune'] * 100).round(1)
    income.columns = ['Categorie venit', 'Nr. clienți', 'Rată daune (%)']
    return income


def get_distributie_vehicul(df):
    """Returneaza distributia pe tip de vehicul."""
    vehicul = df.groupby('VEHICLE_TYPE').agg(
        nr_clienti=('OUTCOME', 'count'),
        rata_daune=('OUTCOME', 'mean')
    ).round(3).reset_index()
    vehicul['rata_daune'] = (vehicul['rata_daune'] * 100).round(1)
    vehicul.columns = ['Tip vehicul', 'Nr. clienți', 'Rată daune (%)']
    return vehicul


def get_married_vs_single(df):
    """Returneaza statistici casatoriti vs necasatoriti."""
    married = df.groupby('MARRIED').agg(
        nr_clienti=('OUTCOME', 'count'),
        rata_daune=('OUTCOME', 'mean'),
        procent=('OUTCOME', lambda x: round(len(x) / len(df) * 100, 1))
    ).round(3).reset_index()
    married['rata_daune'] = (married['rata_daune'] * 100).round(1)
    married['MARRIED'] = married['MARRIED'].map({
        0.0: 'Necăsătorit',
        1.0: 'Căsătorit'
    })
    married.columns = ['Stare civilă', 'Nr. clienți', 'Rată daune (%)', 'Procent (%)']
    return married


def get_profil_risc(df):
    """Returneaza profilul mediu de risc pe categorie outcome."""
    cols_risc = ['SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'CREDIT_SCORE', 'ANNUAL_MILEAGE']
    profil = df.groupby('OUTCOME')[cols_risc].mean().round(2).reset_index()
    profil['OUTCOME'] = profil['OUTCOME'].map({
        0.0: 'Fără daună',
        1.0: 'Cu daună'
    })
    profil.columns = ['Categorie', 'Încălcări viteză (medie)', 'Accidente trecut (medie)',
                      'Credit score (medie)', 'Kilometraj anual (medie)']
    return profil


def get_top_daune_driving_exp(df):
    """Returneaza rata de daune pe experienta la volan."""
    exp = df.groupby('DRIVING_EXPERIENCE').agg(
        nr_clienti=('OUTCOME', 'count'),
        rata_daune=('OUTCOME', 'mean')
    ).round(3).reset_index()
    exp['rata_daune'] = (exp['rata_daune'] * 100).round(1)
    exp.columns = ['Experiență condus', 'Nr. clienți', 'Rată daune (%)']
    return exp.sort_values('Rată daune (%)', ascending=False)