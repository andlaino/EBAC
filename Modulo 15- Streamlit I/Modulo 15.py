import pandas as pd
import streamlit as st

# Título do aplicativo
st.title("Modulo 15  Streamlit")

# Criar um seletor para escolher a página
page = st.selectbox("Escolha a página", ("Página Inicial", "Análise de Sexo", "Análise da Mãe", "Análise de Gravidez", "Mapa"))

# Mostrar as páginas com base na seleção
if page == "Página Inicial":
    st.subheader("__*Tarefa do Modulo 15*__")
    st.markdown("__*Acesse os links abaixo, leia o conteúdo e crie uma aplicação com streamlit reproduzindo pelo menos 20 códigos extraídos das páginas.*__")
    st.markdown("Isso é um link pro streamlit [get started](https://docs.streamlit.io/get-started)")
    st.markdown("Isso é um link pro streamlit [create app](https://docs.streamlit.io/get-started/tutorials/create-an-app)")
    st.markdown("Isso é um link pro streamlit [using](https://docs.streamlit.io/knowledge-base/using-streamlit)")
    st.markdown("Isso é um link pro streamlit [caching](https://docs.streamlit.io/develop/concepts/architecture/caching)")
    st.markdown("Isso é um link pro streamlit [api](https://docs.streamlit.io/develop/api-reference)")
    st.markdown("Isso é um link pro streamlit [session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)")
    st.markdown("Isso é um link pro streamlit [cheat](https://cheat-sheet.streamlit.app/)")
elif page == "Análise de Sexo":
    import pages.sexo  # Importa a página de análise de sexo
elif page == "Análise da Mãe":
    import pages.mae  # Importa a página de Análise da Mãe
elif page == "Análise de Gravidez":
    import pages.gravidez  # Importa a página de análise de gravidez
elif page == "Mapa":
    import pages.mapa  # Importa a página de análise do mapas
