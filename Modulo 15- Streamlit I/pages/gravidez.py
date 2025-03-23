import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt


# Carregar os dados
sinasc = pd.read_csv('input_M15_SINASC_RO_2019.csv')


# Título da página
st.title("Análise de Tipo de Gravidez")

# Verificar se a coluna 'GRAVIDEZ' (Tipo de Gravidez) está disponível
if 'GRAVIDEZ' in sinasc.columns:
    gravidez_counts = sinasc['GRAVIDEZ'].value_counts().reset_index()
    gravidez_counts.columns = ['Tipo de Gravidez', 'Quantidade']
    
    # Mostrar a distribuição dos tipos de gravidez
    st.write(gravidez_counts)
    
        # Criar gráfico com Altair
    chart = alt.Chart(gravidez_counts).mark_bar().encode(
        x='Tipo de Gravidez',
        y='Quantidade',
        color='Tipo de Gravidez'
    ).properties(
        title=""
    )


    # Exibir o gráfico no Streamlit
    st.altair_chart(chart, use_container_width=True)
else:
    st.error("A coluna 'GRAVIDEZ' não está disponível na base de dados.")




st.markdown("----")



sinasc['MES'] = pd.to_datetime(sinasc['DTNASC'], errors='coerce').dt.month

# Filtrar os dados válidos e organizar os partos por mês
monthly_delivery_data = sinasc.groupby(['MES', 'PARTO']).size().unstack(fill_value=0)

# Renomear as colunas para maior clareza
monthly_delivery_data.columns = ['Cesárea', 'Vaginal']
monthly_delivery_data.reset_index(inplace=True)

# Criar o gráfico interativo com cores
fig = px.line(
    monthly_delivery_data,
    x='MES',
    y=['Cesárea', 'Vaginal'],
    markers=True,
    title="Partos Cesárea vs Vaginal por Mês",
    labels={"value": "Quantidade de Partos", "MES": "Mês"},
    color_discrete_map={
        'Cesárea': 'blue',
        'Vaginal': 'orange',
    }
)

# Adicionar interatividade ao gráfico
fig.update_traces(mode="lines+markers")
fig.update_layout(
    hovermode="x unified",
    xaxis=dict(tickmode="linear"),
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)

# Título do aplicativo
st.title("__*Evolução dos Nascimentos Diários*__")

# Converter a coluna DTNASC para o tipo datetime
sinasc['DTNASC'] = pd.to_datetime(sinasc['DTNASC'], errors='coerce')

# Agrupar os nascimentos por dia e contar a quantidade de nascimentos
daily_births = sinasc.groupby(sinasc['DTNASC'].dt.date).size()

# Exibir o gráfico de linha com os dados de nascimentos diários
st.line_chart(daily_births)


