import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def grafic_outcome_bar(df_outcome):
    """Grafic bare pentru distributia outcome."""
    fig = go.Figure(data=[
        go.Bar(
            x=df_outcome['Outcome'],
            y=df_outcome['Nr. clienți'],
            text=df_outcome['Procent (%)'].apply(lambda x: f'{x}%'),
            textposition='outside',
            marker_color=['#2E75B6', '#C55A11']
        )
    ])
    fig.update_layout(
        title='Distribuția clienților: Cu daună vs. Fără daună',
        xaxis_title='Categorie',
        yaxis_title='Număr clienți',
        height=400
    )
    return fig


def grafic_outcome_pie(df_outcome):
    """Grafic pie pentru distributia outcome."""
    fig = go.Figure(data=[
        go.Pie(
            labels=df_outcome['Outcome'],
            values=df_outcome['Nr. clienți'],
            hole=0.3,
            marker_colors=['#2E75B6', '#C55A11']
        )
    ])
    fig.update_layout(title='Proporția clienților cu/fără daună', height=400)
    return fig


def grafic_varsta_bar(df_varsta):
    """Grafic bare pentru distributia pe varsta."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_varsta['Categorie vârstă'],
        y=df_varsta['Nr. clienți'],
        name='Nr. clienți',
        marker_color='#2E75B6',
        yaxis='y',
        offsetgroup=1
    ))
    fig.add_trace(go.Scatter(
        x=df_varsta['Categorie vârstă'],
        y=df_varsta['Rată daune (%)'],
        name='Rată daune (%)',
        marker_color='#C55A11',
        yaxis='y2',
        mode='lines+markers'
    ))
    fig.update_layout(
        title='Distribuția clienților și rata de daune pe categorii de vârstă',
        xaxis_title='Categorie vârstă',
        yaxis=dict(title='Nr. clienți'),
        yaxis2=dict(title='Rată daune (%)', overlaying='y', side='right'),
        height=400,
        legend=dict(x=0.01, y=0.99)
    )
    return fig


def grafic_income_bar(df_income):
    """Grafic bare pentru distributia pe venit."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_income['Categorie venit'],
        y=df_income['Nr. clienți'],
        name='Nr. clienți',
        marker_color='#2E75B6'
    ))
    fig.update_layout(
        title='Distribuția clienților pe categorii de venit',
        xaxis_title='Categorie venit',
        yaxis_title='Număr clienți',
        height=400
    )
    return fig


def grafic_vehicul_bar(df_vehicul):
    """Grafic bare pentru distributia pe tip vehicul."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_vehicul['Tip vehicul'],
        y=df_vehicul['Nr. clienți'],
        text=df_vehicul['Rată daune (%)'].apply(lambda x: f'{x}%'),
        textposition='outside',
        marker_color=['#2E75B6', '#C55A11']
    ))
    fig.update_layout(
        title='Distribuția clienților pe tip de vehicul (cu rata de daune)',
        xaxis_title='Tip vehicul',
        yaxis_title='Număr clienți',
        height=400
    )
    return fig


def grafic_married_bar(df_married, col_y, titlu):
    """Grafic bare pentru comparatie casatoriti vs necasatoriti."""
    fig = go.Figure(data=[
        go.Bar(
            x=df_married['Stare civilă'],
            y=df_married[col_y],
            marker_color=['#2E75B6', '#70AD47']
        )
    ])
    fig.update_layout(
        title=titlu,
        xaxis_title='Stare civilă',
        yaxis_title=col_y,
        height=400
    )
    return fig


def grafic_valori_lipsa(df):
    """Grafic bare pentru valorile lipsa."""
    lipsa = df.isnull().sum()
    lipsa = lipsa[lipsa > 0].sort_values(ascending=False)
    
    if len(lipsa) == 0:
        # Nu exista valori lipsa
        fig = go.Figure()
        fig.add_annotation(
            text="Nu există valori lipsă în acest set de date",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        lipsa_df = pd.DataFrame({'Coloană': [], 'Valori lipsă': [], 'Procent (%)': []})
    else:
        lipsa_df = pd.DataFrame({
            'Coloană': lipsa.index,
            'Valori lipsă': lipsa.values,
            'Procent (%)': (lipsa.values / len(df) * 100).round(1)
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=lipsa_df['Coloană'],
                y=lipsa_df['Valori lipsă'],
                text=lipsa_df['Procent (%)'].apply(lambda x: f'{x}%'),
                textposition='outside',
                marker_color='#C55A11'
            )
        ])
        fig.update_layout(
            title='Valori lipsă per coloană',
            xaxis_title='Coloană',
            yaxis_title='Număr valori lipsă',
            height=400
        )
    
    return fig, lipsa_df


def grafic_outlieri(outlieri_dict):
    """Grafic bare pentru numarul de outlieri per coloana."""
    df_out = pd.DataFrame(list(outlieri_dict.items()), columns=['Coloană', 'Nr. outlieri'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_out['Coloană'],
            y=df_out['Nr. outlieri'],
            marker_color='#C55A11'
        )
    ])
    fig.update_layout(
        title='Număr de outlieri detectați per coloană (metoda IQR)',
        xaxis_title='Coloană',
        yaxis_title='Număr outlieri',
        height=400
    )
    return fig


def grafic_elbow(K_range, inertii):
    """Grafic metoda cotului pentru KMeans."""
    fig = go.Figure(data=[
        go.Scatter(
            x=K_range,
            y=inertii,
            mode='lines+markers',
            marker=dict(size=10, color='#2E75B6'),
            line=dict(width=2, color='#2E75B6')
        )
    ])
    fig.update_layout(
        title='Metoda cotului (Elbow Method) pentru alegerea K',
        xaxis_title='Număr de clustere (K)',
        yaxis_title='Inerție (WCSS)',
        height=400
    )
    return fig


def grafic_scatter_clustere(df_cl, col_x, col_y, titlu, xlabel, ylabel):
    """Grafic scatter pentru vizualizarea clusterelor."""
    fig = px.scatter(
        df_cl, x=col_x, y=col_y, color='Cluster',
        title=titlu,
        labels={col_x: xlabel, col_y: ylabel},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=500)
    return fig


def grafic_distributie_clustere_bar(dist_cl):
    """Grafic bare pentru distributia pe clustere."""
    fig = go.Figure(data=[
        go.Bar(
            x=dist_cl['Cluster'],
            y=dist_cl['Nr. accidente'],
            text=dist_cl['Procent (%)'].apply(lambda x: f'{x}%'),
            textposition='outside',
            marker_color=px.colors.qualitative.Set2[:len(dist_cl)]
        )
    ])
    fig.update_layout(
        title='Distribuția clienților pe clustere',
        xaxis_title='Cluster',
        yaxis_title='Număr clienți',
        height=400
    )
    return fig


def grafic_distributie_clustere_pie(dist_cl):
    """Grafic pie pentru distributia pe clustere."""
    fig = go.Figure(data=[
        go.Pie(
            labels=dist_cl['Cluster'],
            values=dist_cl['Nr. accidente'],
            hole=0.3,
            marker_colors=px.colors.qualitative.Set2[:len(dist_cl)]
        )
    ])
    fig.update_layout(title='Proporția clienților pe clustere', height=400)
    return fig


def grafic_outcome_cluster(df_cl_outcome):
    """Grafic bare pentru rata de daune per cluster."""
    fig = go.Figure(data=[
        go.Bar(
            x=df_cl_outcome['Cluster'],
            y=df_cl_outcome['Rată daune (%)'],
            marker_color='#C55A11'
        )
    ])
    fig.update_layout(
        title='Rata de daune per cluster',
        xaxis_title='Cluster',
        yaxis_title='Rată daune (%)',
        height=400
    )
    return fig


def grafic_confusion_matrix(cm):
    """Heatmap pentru matricea de confuzie."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prezis: Fără daună', 'Prezis: Cu daună'],
        y=['Real: Fără daună', 'Real: Cu daună'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=False
    ))
    fig.update_layout(
        title='Matricea de confuzie',
        xaxis_title='Predicție',
        yaxis_title='Realitate',
        height=400
    )
    return fig


def grafic_coeficienti_logistic(features, coefs):
    """Grafic bare pentru coeficientii regresiei logistice."""
    df_coef = pd.DataFrame({
        'Variabilă': features,
        'Coeficient': coefs
    }).sort_values('Coeficient', key=abs, ascending=False)
    
    colors = ['#C55A11' if x < 0 else '#2E75B6' for x in df_coef['Coeficient']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df_coef['Variabilă'],
            x=df_coef['Coeficient'],
            orientation='h',
            marker_color=colors
        )
    ])
    fig.update_layout(
        title='Importanța variabilelor (coeficienți regresie logistică)',
        xaxis_title='Coeficient',
        yaxis_title='Variabilă',
        height=500
    )
    return fig


def grafic_coeficienti_ols(coef_df):
    """Grafic bare pentru coeficientii OLS."""
    df_plot = coef_df[coef_df['Variabilă'] != 'const'].copy()
    df_plot = df_plot.sort_values('Coeficient', key=abs, ascending=False)
    
    colors = ['#C55A11' if x < 0 else '#2E75B6' for x in df_plot['Coeficient']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df_plot['Variabilă'],
            x=df_plot['Coeficient'],
            orientation='h',
            marker_color=colors,
            text=df_plot['Semnificativ'],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title='Coeficienții modelului OLS (✅ = semnificativ p<0.05)',
        xaxis_title='Coeficient',
        yaxis_title='Variabilă',
        height=500
    )
    return fig


def grafic_reziduale(fitted, residuals):
    """Grafic scatter pentru reziduale."""
    fig = go.Figure(data=[
        go.Scatter(
            x=fitted,
            y=residuals,
            mode='markers',
            marker=dict(color='#2E75B6', opacity=0.5)
        )
    ])
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(
        title='Grafic reziduale',
        xaxis_title='Valori estimate',
        yaxis_title='Reziduale',
        height=400
    )
    return fig


def grafic_distributie_reziduale(residuals):
    """Histogram pentru distributia reziduurilor."""
    fig = go.Figure(data=[
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='#2E75B6'
        )
    ])
    fig.update_layout(
        title='Distribuția reziduurilor',
        xaxis_title='Reziduale',
        yaxis_title='Frecvență',
        height=400
    )
    return fig


def grafic_real_vs_estimat(y_real, y_estimat):
    """Grafic scatter pentru valori reale vs estimate."""
    fig = go.Figure(data=[
        go.Scatter(
            x=y_real,
            y=y_estimat,
            mode='markers',
            marker=dict(color='#2E75B6', opacity=0.5)
        )
    ])
    # Linie diagonala perfecta
    min_val = min(y_real.min(), y_estimat.min())
    max_val = max(y_real.max(), y_estimat.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Predicție perfectă'
    ))
    fig.update_layout(
        title='Valori reale vs. valori estimate',
        xaxis_title='Valori reale',
        yaxis_title='Valori estimate',
        height=400,
        showlegend=True
    )
    return fig