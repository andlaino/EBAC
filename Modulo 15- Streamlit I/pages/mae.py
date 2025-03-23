import pandas as pd
import streamlit as st


# Carregar os dados
sinasc = pd.read_csv('input_M15_SINASC_RO_2019.csv')

# Título da página
st.title("Análise da Mãe")

# Verificar se a coluna 'ESCMAE' (Nível de Escolaridade da Mãe) está disponível
if 'ESCMAE' in sinasc.columns:
    educacao_counts = sinasc['ESCMAE'].value_counts().reset_index()
    educacao_counts.columns = ['Nível de Educação', 'Quantidade']
    
    # Mostrar a distribuição dos níveis de escolaridade
    st.write(educacao_counts)
    
    # Criar um gráfico de barras
    st.bar_chart(educacao_counts.set_index('Nível de Educação')['Quantidade'])
else:
    st.error("A coluna 'ESCMAE' não está disponível na base de dados.")

# Título
st.title("Distribuição de Cor da Mãe por Cidade de Nascimento")

# Verificar se as colunas 'RACACOR' (Cor/Raça da Mãe) e 'munResNome' (Cidade de Nascimento) estão disponíveis
if 'RACACOR' in sinasc.columns and 'munResNome' in sinasc.columns:
    # Obter a lista de cidades para o filtro
    cidades = sinasc['munResNome'].unique()

    # Criar o filtro para a cidade
    cidade_selecionada = st.selectbox("Selecione a Cidade:", cidades)

    # Filtrar os dados pela cidade selecionada
    dados_filtrados = sinasc[sinasc['munResNome'] == cidade_selecionada]

    # Agrupar os dados pela cor/raça da mãe e contar a quantidade de ocorrências
    racacor_cidade = dados_filtrados.groupby(['RACACOR']).size().reset_index(name='Quantidade')

    # Pivotar os dados para que as colunas representem as cores/raças e as linhas representem as quantidades
    pivot_data = racacor_cidade.pivot_table(index='RACACOR', values='Quantidade', fill_value=0)

    # Exibir o gráfico de área
    st.area_chart(pivot_data)

else:
    st.error("As colunas 'RACACOR' (Cor/Raça da Mãe) e 'munResNome' (Cidade de Nascimento) não estão disponíveis na base de dados.")



# Título do aplicativo
st.title("Idade Média da Mãe e do Pai")

# Verificar se as colunas de idade da mãe e do pai estão disponíveis
if 'IDADEMAE' in sinasc.columns and 'IDADEPAI' in sinasc.columns:
    # Calcular as idades médias da mãe e do pai, excluindo valores nulos
    idade_media_mae = sinasc['IDADEMAE'].mean()
    idade_media_pai = sinasc['IDADEPAI'].mean()

    # Criar um DataFrame com as idades médias para plotar
    dados_idade = pd.DataFrame({
        'Categoria': ['Mãe', 'Pai'],
        'Idade Média': [idade_media_mae, idade_media_pai]
    })

    # Exibir os dados das idades médias
    st.write(dados_idade)

    # Criar o gráfico de barras
    st.bar_chart(dados_idade.set_index('Categoria')['Idade Média'])

else:
    st.error("As colunas 'IDADEMAE' (Idade da Mãe) e 'IDADEPAI' (Idade do Pai) não estão disponíveis na base de dados.")





