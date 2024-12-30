import streamlit as st

# ==================================================================
# Configurações
# ==================================================================
st.set_page_config(page_title = 'Home',
                  layout= 'centered')

# ==================================================================
# Barra Lateral
# ==================================================================
#st.sidebar.markdown( '# Celpe-Bras 2015/2' ) 
#st.sidebar.markdown( """---""" )

# ==================================================================
# Layout
# ==================================================================

st.header('Celpe-Bras 2015/2')

st.subheader('Um projeto de Visual Analytics', divider='gray')

st.markdown("""
           A partir do conhecimento obtido na disciplina de Visual Analytics, foram desenvolvidos os dashboards a seguir. Assim, aqui você encontrará uma análise realizada a partir dos textos da Tarefa 4 do exame Celpe-Bras do semestre 2015/2. 
           
           Abaixo você encontra enunciado e texto de apoio referente a essa tarefa.
"""
)

image_path = "tarefa4.png"
st.image(image_path, use_column_width=True) # , caption="This is an example PNG image", #, use_column_width=True

st.markdown("""
           Em Tokens e Types você obterá insights acerca do número de tokens e types dos textos de acordo com cada nota.
           
           Em Keywords você visualizará as principais palavras encontradas nos textos de cada nota.

           Em Embeddings você visualizará o gráfico t-SNE para as embeddings fixas do corpus.
"""
)

st.markdown('Dados não estão publicamente disponíveis.')