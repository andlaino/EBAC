import pandas as pd
import streamlit as st

# Carregar os dados
sinasc = pd.read_csv('input_M15_SINASC_RO_2019.csv')

# Título da página
st.title("Análise de Sexo dos Bebês")

# Verificar se a coluna 'SEXO' está disponível
if 'SEXO' in sinasc.columns:
    sexo_counts = sinasc['SEXO'].value_counts().reset_index()
    sexo_counts.columns = ['Sexo', 'Quantidade']
    
    # Mostrar a distribuição dos sexos
    st.write(sexo_counts)
    
    # Criar um gráfico de barras
    st.bar_chart(sexo_counts.set_index('Sexo')['Quantidade'])
else:
    st.error("A coluna 'SEXO' não está disponível na base de dados.")


# Título e descrição
st.title("Peso Médio ao Nascer por Sexo do Bebê")
st.markdown("Este gráfico de progresso apresenta o peso médio ao nascer por sexo dos bebês com base nos dados de nascimentos.")

# Processar os dados para calcular o peso médio por sexo
peso_medio_sexo = sinasc.groupby('SEXO')['PESO'].mean().reset_index()
peso_medio_sexo.columns = ['Sexo', 'Peso Médio ao Nascer']

# Mapear os códigos de sexo para rótulos mais claros (se necessário)
peso_medio_sexo['Sexo'] = peso_medio_sexo['Sexo'].replace({1: 'Masculino', 2: 'Feminino'})

# Exibir barras de progresso para cada sexo
for _, row in peso_medio_sexo.iterrows():
    sexo = row['Sexo']
    peso_medio = row['Peso Médio ao Nascer']
    progress = min(int(peso_medio / 60), 100)  # Normalizar o peso em uma escala de 0 a 100 para o progresso
    st.subheader(f"{sexo}: {peso_medio:.2f}g")
    st.progress(progress)


