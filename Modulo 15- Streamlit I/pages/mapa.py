import pandas as pd
import streamlit as st

# Carregar os dados
sinasc = pd.read_csv('input_M15_SINASC_RO_2019.csv')

st.title("Mapa Interativo de Municípios com Filtro")

# Verificar se as colunas de latitude e longitude estão disponíveis
if 'munResLat' in sinasc.columns and 'munResLon' in sinasc.columns:
    # Criar uma lista de municípios únicos
    municipios = sinasc['munResNome'].dropna().unique()
    
    # Criar um filtro para seleção de municípios
    municipio_selecionado = st.multiselect(
        "Selecione o(s) município(s):",
        options=sorted(municipios),
        default=sorted(municipios[:5])  # Seleciona os 5 primeiros por padrão
    )
    
    # Filtrar os dados com base no município selecionado
    dados_filtrados = sinasc[sinasc['munResNome'].isin(municipio_selecionado)]
    map_data = dados_filtrados[['munResLat', 'munResLon']].dropna()
    map_data.columns = ['lat', 'lon']  # Renomear para compatibilidade com st.map
    
    # Calcular a quantidade de nascimentos por município
    qtd_nascimentos = dados_filtrados['munResNome'].value_counts()
    
    # Mostrar a quantidade de nascimentos por município
    st.subheader("Quantidade de Nascimentos por Município")
    st.write(qtd_nascimentos)

    # Exibir o mapa
    st.subheader(f"Mapa para os Municípios Selecionados ({len(map_data)} pontos)")
    st.map(map_data)
    

else:
    st.error("As colunas de latitude ('munResLat') e longitude ('munResLon') não estão disponíveis na base de dados.")




# Título e descrição
st.title("Top 10 Municípios com Mais Nascimentos")
st.markdown("Este gráfico de linha apresenta os 10 municípios com mais nascimentos registrados com base nos dados.")

# Processar os dados
top_municipios = sinasc['munResNome'].value_counts().head(10).reset_index()
top_municipios.columns = ['Município', 'Frequência']

# Criar um DataFrame para o gráfico de linha
line_chart_data = pd.DataFrame({
    'Município': top_municipios['Município'],
    'Frequência': top_municipios['Frequência']
}).set_index('Município')

# Exibir o gráfico de linha no Streamlit
st.line_chart(line_chart_data)


# Título e descrição
st.title("Top 10 Municípios com Menos Nascimentos")
st.markdown("Este gráfico de linha apresenta os 10 municípios com menos nascimentos registrados com base nos dados.")
# Processar os dados para encontrar os municípios com menos nascimentos
bottom_municipios = sinasc['munResNome'].value_counts().tail(10).reset_index()
bottom_municipios.columns = ['Município', 'Frequência']

# Criar um DataFrame para o gráfico de linha
line_chart_data = pd.DataFrame({
    'Município': bottom_municipios['Município'],
    'Frequência': bottom_municipios['Frequência']
}).set_index('Município')

# Exibir o gráfico de linha no Streamlit
st.line_chart(line_chart_data)