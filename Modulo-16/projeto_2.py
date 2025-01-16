import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from ydata_profiling import ProfileReport

from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


st.markdown("<h1 style='text-align: center; color: black;'>Previsão de renda</h1>", unsafe_allow_html=True)

renda = pd.read_csv('C:/Users/ander/OneDrive/Documentos/EBAC/previsao_renda/input/previsao_de_renda.csv')

st.markdown("----")
def intro():
    st.markdown('**<span style="font-size:28px;">Etapa 1 CRISP - DM: Entendimento do negócio</span>**', unsafe_allow_html=True)
    st.write("A previsão de renda é útil pra diversas instituições financeiras, tal informação pode guiar em desisões como o valor máximo de empréstimos concedidos, limite de crédito para um cliente e também a oferta de classes de cartões para os clientes. Utilizando a base de dados que contém as informações de cadastro de outros clientes, é possível realizar a análise de como tais informações se relacionam com a renda apresentada para assim gerar previsões para futuros clientes.")
    st.write("Este projeto tem como objetivo realizar a análise dos dados e criar um modelo estatistico para a previsão de renda.")
    st.markdown('**<span style="font-size:28px;">Etapa 2 Crisp-DM: Entendimento dos dados</span>**', unsafe_allow_html=True)
    st.write("A base de dados utilizada neste projeto foi disponibilizada pela EBAC no curso de Ciência de Dados.")
    st.markdown('**<span style="font-size:20px;">Dicionário de dados</span>**', unsafe_allow_html=True)
    st.write("Tabela que descreve as variáveis a serem trabalhadas:")
    st.markdown(''' | Variável | Descrição | Tipo |
| :---:         |     :---:      |          :---: |
| data_ref  | Data de referência     | Objeto    |
| id_cliente     | Id do cliente       | Int      |
| sexo                    |  Sexo do cliente                                    | Objeto |
| posse_de_veiculo        |  Possui veiculo?                                    | Booleano |
| posse_de_imovel         |  Possui imovel?                                     | Booleano |
| qtd_filhos              |  Quantos filhos?                                    | Int |
| tipo_renda              |  Tipo de renda                                      | Objeto |
| educacao                |  Escolaridade                                       | Objeto |
| estado_civil            |  Estado civil                                       | Objeto |
| tipo_residencia         |  Tipo de residência                                 | Objeto |
| idade                   |  Idade                                              | Int |
| tempo_emprego           |  Tempo de empregabiilidade                          | Float |
| qt_pessoas_residencia   |  Quantidade de pessoas na residência                | Float |
| renda                   |  Renda                                              | Float |
            ''')
    st.markdown('**<span style="font-size:20px;">Carregando os pacotes</span>**', unsafe_allow_html=True)
    st.write('É considerado uma boa prática carregar os pacotes que serão utilizados como a primeira coisa do programa.')
    code = '''import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import tree

%matplotlib inline'''
    st.code(code, language="python")
    st.markdown('**<span style="font-size:20px;">Carregando os dados</span>**', unsafe_allow_html=True)
    st.write("O comando pd.read_csv é um comando da biblioteca pandas (pd.) e carrega os dados do arquivo csv indicado para um objeto dataframe do pandas.")
    code2 = '''renda = pd.read_csv('./input/previsao_de_renda.csv')'''
    st.code(code2, language=None)
    code3 = '''renda.drop(labels='Unnamed: 0', axis = 1, inplace=True)
renda'''
    st.code(code3, language=None)

    
    renda.drop(labels='Unnamed: 0', axis = 1, inplace=True)
    renda
    code4 = '''renda.dtypes'''
    st.code(code4, language=None)
    renda.dtypes
    st.markdown('**<span style="font-size:20px;">Entendimento dos dados - Univariada</span>**', unsafe_allow_html=True)
    st.write("Nesta etapa tipicamente avaliamos a distribuição de todas as variáveis.")
    code5 = '''prof = ProfileReport(renda, explorative=True, minimal=True)
prof'''
    st.code(code5, language=None)
    #prof = ProfileReport(renda, explorative=True, minimal=True)
    #prof
    #prof.to_file('C:/Users/ander/OneDrive/Documentos/EBAC/previsao_renda/output/renda_analisys.html')
    code6 = '''prof.to_file('./output/renda_analisys.html')'''
    st.code(code6, language=None)
    st.markdown('**<span style="font-size:12px;">Os Resultados Univariada, da pagina html que foi gerada está nesse [link](C:/Users/ander/OneDrive/Documentos/EBAC/previsao_renda/output/renda_analisys.html)</span>**', unsafe_allow_html=True)



def Resultado_Univariada():



    

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

    @st.cache_data
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def plotting_demo():



    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def data_frame_demo():



    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

page_names_to_funcs = {
    "—": intro,
    "Plotting Demo": plotting_demo,
    "Resultado Univariada": Resultado_Univariada,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()