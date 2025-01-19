# Importando as bibliotecas necessárias
import streamlit as st  # Biblioteca para criar aplicações web interativas
import pandas as pd  # Manipulação de dados
import seaborn as sns  # Visualização de dados
import numpy as np  # Operações numéricas
import matplotlib.pyplot as plt  # Criação de gráficos
from ydata_profiling import ProfileReport  # Geração de relatórios exploratórios automáticos
from sklearn.ensemble import RandomForestRegressor  # Modelo de regressão
from sklearn.model_selection import train_test_split, GridSearchCV  # Divisão de dados e ajuste de hiperparâmetros
from sklearn.metrics import mean_squared_error  # Métrica de avaliação de modelo
from sklearn import tree  # Algoritmo de árvore de decisão
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io  # Manipulação de fluxos de entrada e saída
import re  # Operações com expressões regulares
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# Configurações iniciais do Streamlit
st.set_page_config(
    page_title="Projeto Interativo com Streamlit",
    page_icon="💰",
    layout="wide",
)

# Título do aplicativo no Streamlit
st.markdown("<h1 style='text-align: center; color: black;'>Previsão de renda</h1>", unsafe_allow_html=True)

# Carregando o conjunto de dados
renda = pd.read_csv('C:/Users/ander/OneDrive/Documentos/EBAC/Modulo-16/input/previsao_de_renda.csv')

# Linha divisória para organização visual no Streamlit
st.markdown("----")

# Função de introdução e descrição do projeto
def intro():

    st.markdown('<div style="text-align: center;"><span style="font-size:36px;font-weight: bold;">📝 Exercício Completo</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center;"><span style="font-size:24px;">Esta seção permite realizar o exercício completo, onde você posso analisar dados financeiros e interagir com gráficos.</span>', unsafe_allow_html=True)
    

   



    

       
    
    

   
    
    

def Resultado_Univariada():


    

    renda = pd.read_csv('C:/Users/ander/OneDrive/Documentos/EBAC/Modulo-16/input/previsao_de_renda.csv')

    st.write('# Análise exploratória da previsão de renda')





    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader('Quantidade de pessoas por sexo')
            plt.figure(figsize=(1, 2))
            fig = px.histogram(renda, x='sexo')
            st.plotly_chart(fig)

        with col2:
            st.subheader('Escolaridade')
            plt.figure(figsize=(0.5, 0.5))
            fig = px.pie(renda, names='educacao')
            st.plotly_chart(fig)

     
        with col3:
            st.subheader('Quantidade de pessoas que vivem em uma mesma residência')
            plt.figure(figsize=(1, 2))
            fig = px.histogram(renda, x='qt_pessoas_residencia')
            st.plotly_chart(fig)

    with st.container():
        col1, col2 = st.columns([1, 1])

        # Gráfico 1: Média de Renda por Idade
        with col1:
            st.subheader('Média de Renda por Idade')

            if renda.empty:
                st.warning("Nenhum dado disponível para a faixa de datas selecionada.")
            else:
                # Calcular média de renda por idade
                media_renda_por_idade = renda.groupby('idade')['renda'].mean().reset_index()

                # Verificar se o resultado do agrupamento é válido
                if media_renda_por_idade.empty:
                    st.warning("Não há dados suficientes para calcular a média de renda por idade.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=media_renda_por_idade['idade'],
                        y=media_renda_por_idade['renda'],
                        marker_color='#87CEEB'
                    ))

                    # Adicionar título e rótulos aos eixos
                    fig.update_layout(
                        title="Média de Renda por Idade",
                        xaxis_title='Idade',
                        yaxis_title='Média de Renda',
                        xaxis=dict(tickmode='array'),
                        template='plotly_white'
                    )

                    st.plotly_chart(fig)

    # Gráfico 2: Tipo de Renda
    with col2:
        st.subheader('Tipo de Renda')

        if 'tipo_renda' not in renda.columns or renda['tipo_renda'].isnull().all():
            st.warning("A coluna 'tipo_renda' está ausente ou sem valores.")
        else:
            fig = px.histogram(renda, x='tipo_renda')
            st.plotly_chart(fig)




    

  
def plotting_demo():

    # Explicando a etapa 1 do CRISP-DM
    st.markdown('**<span style="font-size:28px;">Etapa 1 CRISP - DM: Entendimento do negócio</span>**', unsafe_allow_html=True)
    st.write("A previsão de renda é útil para diversas instituições financeiras, auxiliando em decisões como o valor máximo de empréstimos, limites de crédito e oferta de produtos financeiros.")
    st.write("O objetivo deste projeto é analisar os dados e criar um modelo estatístico para previsão de renda.")

    # Explicando a etapa 2 do CRISP-DM
    st.markdown('**<span style="font-size:28px;">Etapa 2 Crisp-DM: Entendimento dos dados</span>**', unsafe_allow_html=True)
    st.write("A base de dados foi disponibilizada pela EBAC no curso de Ciência de Dados.")
    st.markdown('**<span style="font-size:20px;">Dicionário de dados</span>**', unsafe_allow_html=True)
    st.write("Tabela que descreve as variáveis do dataset:")
    st.markdown(''' 
    | Variável               | Descrição                                      | Tipo      |
    |------------------------|-----------------------------------------------|-----------|
    | data_ref               | Data de referência                           | Objeto    |
    | id_cliente             | Identificador do cliente                     | Int       |
    | sexo                   | Sexo do cliente                              | Objeto    |
    | posse_de_veiculo       | Possui veículo?                              | Booleano  |
    | posse_de_imovel        | Possui imóvel?                               | Booleano  |
    | qtd_filhos             | Quantidade de filhos                         | Int       |
    | tipo_renda             | Tipo de renda                                | Objeto    |
    | educacao               | Escolaridade                                 | Objeto    |
    | estado_civil           | Estado civil                                 | Objeto    |
    | tipo_residencia        | Tipo de residência                           | Objeto    |
    | idade                  | Idade                                        | Int       |
    | tempo_emprego          | Tempo de empregabilidade                     | Float     |
    | qt_pessoas_residencia  | Quantidade de pessoas na residência          | Float     |
    | renda                  | Renda                                        | Float     |
    ''')

    # Carregando os pacotes utilizados
    st.markdown('**<span style="font-size:20px;">Carregando os pacotes</span>**', unsafe_allow_html=True)
    code = '''# Importando as bibliotecas necessárias
import streamlit as st  # Biblioteca para criar aplicações web interativas
import pandas as pd  # Manipulação de dados
import seaborn as sns  # Visualização de dados
import numpy as np  # Operações numéricas
import matplotlib.pyplot as plt  # Criação de gráficos
from ydata_profiling import ProfileReport  # Geração de relatórios exploratórios automáticos
from sklearn.ensemble import RandomForestRegressor  # Modelo de regressão
from sklearn.model_selection import train_test_split, GridSearchCV  # Divisão de dados e ajuste de hiperparâmetros
from sklearn.metrics import mean_squared_error  # Métrica de avaliação de modelo
from sklearn import tree  # Algoritmo de árvore de decisão
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io  # Manipulação de fluxos de entrada e saída
import re  # Operações com expressões regulares
import statsmodels.api as sm'''
    st.code(code, language="python")

    # Carregando os dados
    st.markdown('**<span style="font-size:20px;">Carregando os dados</span>**', unsafe_allow_html=True)
    code2 = '''renda = pd.read_csv('./input/previsao_de_renda.csv')'''
    st.code(code2, language=None)

    # Removendo coluna desnecessária
    code3 = '''renda.drop(labels='Unnamed: 0', axis=1, inplace=True)
renda'''
    renda.drop(labels='Unnamed: 0', axis=1, inplace=True)
    renda

    # Verificando os tipos de dados
    code4 = '''renda.dtypes'''
    st.code(code4, language=None)
    renda.dtypes

    # Análise univariada
    st.markdown('**<span style="font-size:20px;">Entendimento dos dados - Univariada</span>**', unsafe_allow_html=True)
    code5 = '''prof = ProfileReport(renda, explorative=True, minimal=True)
prof'''
    st.code(code5, language=None)

    # Gerando relatório em HTML
    code6 = '''prof.to_file('./output/renda_analisys.html')'''
    st.code(code6, language=None)

    # Análise bivariada
    st.markdown('**<span style="font-size:20px;">Entendimento dos dados - Bivariadas</span>**', unsafe_allow_html=True)
    st.markdown('**<span style="font-size:16px;">Matriz de correlação</span>**', unsafe_allow_html=True)
    plot = sns.pairplot(
        data=renda, 
        hue="tipo_renda", 
        hue_order=['Assalariado', 'Bolsista', 'Empresário', 'Pensionista', 'Servidor público'],
        vars=['tempo_emprego', 'idade', 'qtd_filhos', 'posse_de_imovel', 'qt_pessoas_residencia', 'renda'], 
        markers=["o", "s", "D"]
    )
    st.pyplot(plot.fig)
    st.write('Ao analisar o pairplot, que consiste na matriz de dispersão, é possível identificar alguns outliers na variável renda, os quais podem afetar o resultado da análise de tendência, apesar de ocorrerem com baixa frequência. Além disso, é observada uma baixa correlação entre praticamente todas as variáveis quantitativas, reforçando os resultados obtidos na matriz de correlação.')


    # Selecionar apenas as colunas numéricas
    renda_num = list(renda.select_dtypes('number'))

    # Verifica se as colunas a serem removidas existem e remove
    colunas_para_remover = ['Unnamed: 0', 'id_cliente']
    colunas_para_remover = [col for col in colunas_para_remover if col in renda.columns]  # Verifica se as colunas existem


    st.markdown('**<span style="font-size:16px;">Heatmap</span>**', unsafe_allow_html=True)
    # Criar o gráfico de heatmap
    plt.figure(figsize=(10, 10))
    plt.title('Heatmap de Correlação das Variáveis Numéricas')

    # Gerar e exibir o heatmap de correlação, removendo as colunas indesejadas
    sns.heatmap(renda[renda_num].drop(colunas_para_remover, axis=1).corr(),
            annot=True, 
            center=0, 
            cmap='crest')

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)
    st.write('Com o heatmap, é possível reforçar novamente os resultados de baixa correlação com a variável renda.')


    # Plotando o gráfico de dispersão com a linha de tendência
    st.markdown('**<span style="font-size:16px;">Gráfico de Dispersão com Linha de Tendência</span>**', unsafe_allow_html=True)

    plt.figure(figsize=(16,9))

    # Gráfico de dispersão com tamanho dos pontos proporcional à idade
    sns.scatterplot(x='tempo_emprego',
                    y='renda', 
                    hue='tipo_renda', 
                    size='idade',
                    data=renda,
                    alpha=0.4)

    # Adicionando linha de tendência
    sns.regplot(x='tempo_emprego', 
               y='renda', 
                data=renda, 
                scatter=False, 
                color='.3')

    # Exibindo o gráfico no Streamlit
    st.pyplot(plt)

    st.markdown('**<span style="font-size:16px;">Análise da distribuição da variável renda em função da variável tempo_emprego:</span>**', unsafe_allow_html=True)
    plt.figure(figsize=(16, 9))
    sns.scatterplot(data=renda, x='tempo_emprego', y='renda', hue='sexo')
    plt.title('Distribuição da Renda em Função do Tempo de Emprego por Sexo')
    st.pyplot(plt)

    st.markdown('**<span style="font-size:16px;">Análise das variáveis categóricas em relação a renda:</span>**', unsafe_allow_html=True)
    # Calcular a média da renda
    renda_media = renda['renda'].mean()

    # Exibindo os gráficos com dois gráficos na parte superior e dois na parte inferior
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    
    
    # Gráfico 1: Renda Média por Nível de Educação
    sns.barplot(x="educacao", y="renda", data=renda, ax=axes[0, 0], hue="educacao", legend=False)
    axes[0, 0].set_title("Renda Média por Nível de Educação")
    axes[0, 0].tick_params(axis='x', rotation=90)
    


    # Gráfico 1: Renda Média por Estado Civil
    sns.barplot(x="estado_civil", y="renda", data=renda, ax=axes[0, 1], hue="estado_civil", legend=False)
    axes[0, 1].set_title("Renda Média por Estado Civil")
    axes[0, 1].tick_params(axis='x', rotation=90)
    
    # Gráfico 3: Renda Média por Tipo de Renda
    sns.barplot(x="tipo_renda", y="renda", data=renda, ax=axes[1, 1], hue="tipo_renda", legend=False)
    axes[1, 0].set_title("Renda Média por Tipo de Renda")
    axes[1, 0].tick_params(axis='x', rotation=90)

    # Gráfico 4: Renda Média por Tipo de Residência
    sns.barplot(x="tipo_residencia", y="renda", data=renda, ax=axes[1, 0], hue="tipo_residencia", legend=False)
    axes[1, 1].set_title("Renda Média por Tipo de Residência")
    axes[1, 1].tick_params(axis='x', rotation=90)
    

    
    

    # Ajustar o layout para que os gráficos não sobreponham
    plt.tight_layout()

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)




    # 3 Crisp-DM: Preparação dos dados
    st.markdown('**<span style="font-size:28px;">Etapa 3 Crisp-DM: Preparação dos dados</span>**', unsafe_allow_html=True)
    renda.drop(columns=['data_ref', 'id_cliente'], axis=1, inplace=True)

    # Verificando valores ausentes
    renda_isna = renda.isna().value_counts()

    # Mapeando a coluna 'sexo' para valores numéricos
    renda['sexo'] = renda['sexo'].map({'M': 1, 'F': 0})

    # Criando variáveis dummies para colunas categóricas
    renda_dummies = pd.get_dummies(renda, dtype=int)

    # Exibindo informações do DataFrame
    buffer = io.StringIO()
    renda_dummies.info(buf=buffer)
    info_str = buffer.getvalue()
    st.code(info_str, language='plaintext')

    # Exibindo as primeiras linhas do DataFrame
    st.write(renda_dummies.head())
    
    # Calcular a correlação entre as variáveis dummies e 'renda'
    corr_renda = (renda_dummies.corr()['renda']
                  .sort_values(ascending=False)
                  .to_frame()
                  .reset_index()
                  .rename(columns={'index': 'var', 'renda': 'corr'})
                  .style.bar(color=['red', 'lightgreen'], align=0))

    # Exibir a correlação das variáveis com 'renda'
    st.subheader("Correlação das variáveis com 'renda'")
    st.write(corr_renda)


    
    # Etapa 4 Crisp-DM: Modelagem
    st.markdown('**<span style="font-size:28px;">Etapa 4 Crisp-DM: Modelagem</span>**', unsafe_allow_html=True)
    st.write("Nessa etapa que realizaremos a construção do modelo. Os passos típicos são:")
    st.write("- Selecionar a técnica de modelagem")
    st.write("- Desenho do teste")
    st.write("- Avaliação do modelo")

    # Convertendo colunas categóricas em variáveis dummies
    categorical_columns = renda.select_dtypes(include=['object']).columns.tolist()
    renda_dummies = pd.get_dummies(renda, columns=categorical_columns, drop_first=True)

    # Definindo as variáveis X e y
    X = renda_dummies.drop(['renda'], axis=1).copy()
    y = renda_dummies['renda']

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Treinamento do Modelo 1: Random Forest com profundidade máxima 2
    modelo_1 = RandomForestRegressor(max_depth=2, random_state=42)
    modelo_1.fit(X_train, y_train)

    # Treinamento do Modelo 2: Random Forest com profundidade máxima 8
    modelo_2 = RandomForestRegressor(max_depth=8, random_state=42)
    modelo_2.fit(X_train, y_train)

    # Calculando o R² para os dois modelos
    mse1 = modelo_1.score(X_test, y_test)
    mse2 = modelo_2.score(X_test, y_test)

    st.write(f"O R² do modelo 1 (max_depth=2) é {mse1:.4f}")
    st.write(f"O R² do modelo 2 (max_depth=8) é {mse2:.4f}")

    # Treinamento e visualização do modelo de DecisionTreeRegressor
    st.write("Treinando e visualizando um modelo de DecisionTreeRegressor...")
    modelo_dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=42)
    modelo_dt.fit(X_train, y_train)
    st.markdown('**Ajustando o Modelo 2 com parâmetros adicionais**')

    st.markdown('**<span style="font-size:20px;">Rodando o modelo</span>**', unsafe_allow_html=True)
    st.markdown('**<span style="font-size:16px;">Visualização gráfica da árvore com plot_tree</span>**', unsafe_allow_html=True)

    # Avaliação do modelo
    r2_dt = modelo_dt.score(X_test, y_test)
    st.write(f"O R² do modelo DecisionTreeRegressor (max_depth=8, min_samples_leaf=4) é {r2_dt:.4f}")

    # Visualizando a árvore de decisão
    st.write("Visualização da Árvore de Decisão:")
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(modelo_dt, feature_names=X.columns, filled=True)
    plt.title("Árvore de Decisão (max_depth=8)", fontsize=14)
    st.pyplot(fig)

    st.markdown('**<span style="font-size:16px;">Teste da melhor profundidade e do número mínimo de amostras por folha da árvore:</span>**', unsafe_allow_html=True)

    # Otimização de parâmetros (max_depth e min_samples_leaf)
    r2s = []
    i_indicador = []
    j_indicador = []

    st.write("Iniciando otimização de parâmetros...")
    for i in range(1, 9):
        for j in range(1, 9):
            modelo_2 = RandomForestRegressor(max_depth=i, min_samples_leaf=j, random_state=42)
            modelo_2.fit(X_train, y_train)
            r2_1 = modelo_2.score(X_test, y_test)
            r2s.append(r2_1)
            i_indicador.append(i)
            j_indicador.append(j)

    # Criando o DataFrame para R²
    renda_r2 = pd.DataFrame({'r2': r2s, 'profundidade': i_indicador, 'n_minimo': j_indicador})

    # Criando a matriz de calor (heatmap)
    st.write("Matriz de Calor (Heatmap) dos Parâmetros R²:")
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_table = renda_r2.pivot(index='profundidade', columns='n_minimo', values='r2')
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.2f', ax=ax)
    ax.set_title('Matriz de Calor: R² por Profundidade e n_minimo')

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.markdown('*Configuração para Grid Search*')
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 1000],
    }   

    # Crie o modelo
    rf = RandomForestRegressor(max_depth=7, min_samples_leaf=7, random_state=42)

    # Realize o grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Melhor modelo
    best_model = grid_search.best_estimator_

    # Exibindo o melhor modelo no Streamlit
    st.write("Melhor modelo encontrado pelo GridSearchCV:")
    st.write(best_model)

    st.markdown('*Ajustando o modelo final com os melhores parâmetros*')  
    modelo_final = RandomForestRegressor(max_depth=7, min_samples_leaf=7, n_estimators=500, random_state=42)
    modelo_final.fit(X_train, y_train)

    # Previsões do modelo final
    y_pred = modelo_final.predict(X_test)

    # Exibindo o R² do modelo final no Streamlit
    r2_final = modelo_final.score(X_test, y_test)
    st.write(f"O R² do modelo final (max_depth=7, min_samples_leaf=7, n_estimators=500) é {r2_final:.4f}")

    # Exibindo os primeiros resultados de previsões no Streamlit
    resultados = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
    st.write("Primeiros resultados das previsões:", resultados.head(10))

    index = ['r_squared', 'mean_absolute_error', 'root_mean_squared_error']

    st.markdown('*hiperpametros*')  
    hiperpametros = pd.DataFrame({'conjunto_treino': r2_score(y_train, modelo_final.predict(X_train))}, index=index)
    hiperpametros.loc['mean_absolute_error', 'conjunto_treino'] = mean_absolute_error(y_train, modelo_final.predict(X_train))
    hiperpametros.loc['root_mean_squared_error', 'conjunto_treino'] = mean_squared_error(y_train, modelo_final.predict(X_train)) ** 0.5
    hiperpametros['conjunto_teste'] = r2_score(y_test, modelo_final.predict(X_test))
    hiperpametros.loc['mean_absolute_error', 'conjunto_teste'] = mean_absolute_error(y_test, modelo_final.predict(X_test))
    hiperpametros.loc['root_mean_squared_error', 'conjunto_teste'] = mean_squared_error(y_test, modelo_final.predict(X_test)) ** 0.5

    # Exibindo as métricas no Streamlit
    st.write("Métricas de Avaliação do Modelo:")
    st.write(hiperpametros)


    # Etapa 5 Crisp-DM: Avaliação
    st.markdown('**<span style="font-size:28px;">Etapa 5 Crisp-DM: Avaliação dos resultados</span>**', unsafe_allow_html=True)
    

     # Criando o template de exibição
    template = 'O coeficiente de determinação (𝑅²) da árvore com profundidade = {0} para a base de {1} é: {2:.2f}'

    # Exibindo os resultados no Streamlit
    st.subheader(f"Avaliação da árvore de regressão (𝑅²)")

    # Exibir resultados para treino e teste
    st.write(template.format(modelo_dt.get_depth(), 'treino', modelo_dt.score(X_train, y_train)).replace(".", ","))
    st.write(template.format(modelo_dt.get_depth(), 'teste', r2_dt).replace(".", ","), '\n')

    # Adicionando previsões ao DataFrame
    renda['renda_predict'] = np.round(modelo_dt.predict(X), 2)

    # Exibindo as colunas 'renda' e 'renda_predict'
    st.write("Previsões do modelo DecisionTreeRegressor para a variável 'renda':")
    st.write(renda[['renda', 'renda_predict']])
    

    # Etapa 6 Crisp-DM: Implantação
    st.markdown('**<span style="font-size:28px;">Etapa 6 Crisp-DM: Implantação</span>**', unsafe_allow_html=True)
    st.markdown('Nessa etapa colocamos em uso o modelo desenvolvido, normalmente implementando o modelo desenvolvido em um motor que toma as decisões com algum nível de automação.')
    st.markdown('*Simulando a previsão de renda**')

    # Formulário para entrada de dados
    sexo = st.selectbox('Sexo', ['M', 'F'])
    posse_veiculo = st.selectbox('Posse de Veículo', ['Sim', 'Não'])
    posse_imovel = st.selectbox('Posse de Imóvel', ['Sim', 'Não'])
    qtd_filhos = st.number_input('Quantidade de Filhos', min_value=0, max_value=10, step=1)
    tipo_renda = st.selectbox('Tipo de Renda', ['Assalariado', 'Autônomo', 'Empreendedor', 'Aposentado'])
    educacao = st.selectbox('Educação', ['Superior completo', 'Superior incompleto', 'Médio completo', 'Médio incompleto', 'Fundamental'])
    estado_civil = st.selectbox('Estado Civil', ['Solteiro', 'Casado', 'Divorciado', 'Viúvo'])
    tipo_residencia = st.selectbox('Tipo de Residência', ['Casa', 'Apartamento'])
    idade = st.number_input('Idade', min_value=18, max_value=100, step=1)
    tempo_emprego = st.number_input('Tempo de Emprego (anos)', min_value=0, max_value=50, step=1)
    qt_pessoas_residencia = st.number_input('Número de Pessoas na Residência', min_value=1, max_value=10, step=1)

    # Criando o DataFrame com os dados de entrada
    entrada = pd.DataFrame([{
        'sexo': sexo,
        'posse_de_veiculo': posse_veiculo == 'Sim', 
        'posse_de_imovel': posse_imovel == 'Sim', 
        'qtd_filhos': qtd_filhos, 
        'tipo_renda': tipo_renda, 
        'educacao': educacao, 
        'estado_civil': estado_civil, 
        'tipo_residencia': tipo_residencia, 
        'idade': idade, 
        'tempo_emprego': tempo_emprego, 
        'qt_pessoas_residencia': qt_pessoas_residencia
    }])

    # Convertendo as variáveis categóricas em dummies
    entrada_dummies = pd.get_dummies(entrada)

    # Garantir que a entrada tenha as mesmas colunas de 'X' (variáveis de treino)
    entrada_dummies = entrada_dummies.reindex(columns=X.columns, fill_value=0)

    # Previsão de renda com o modelo de regressão da árvore de decisão
    renda_estimacao = modelo_dt.predict(entrada_dummies).item()

    # Exibindo a previsão de renda
    st.write(f"Renda estimada: R${np.round(renda_estimacao, 2):,.2f}")

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    

    

page_names_to_funcs = {
    "Home": intro,
    "Projeto 2": plotting_demo,
    "Graficos Projeto 2": Resultado_Univariada
    
}

demo_name = st.sidebar.selectbox("Menu", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()