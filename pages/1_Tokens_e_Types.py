import pandas as pd
import numpy as np
import nltk
from collections import Counter #uso para n-gramas
import re # talvez regex vai ajudar na normalizaçao
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==================================================================
# Configurações da Página 
# ==================================================================
st.set_page_config(page_title = 'Tokens e Types',
                  layout= 'centered')

# ==================================================================
#Import dataset
# ==================================================================
csv_file_path = 'corpus_completo.csv'

# Lendo o csv como um df
df = pd.read_csv(csv_file_path)

#Criando uma cópia
df_va = df.copy()

# ==================================================================
# Pré-Processamento
# ==================================================================
df_va.groupby('nota').count()

df_chart = pd.DataFrame()

# Número de textos exposto como variável visto que usarei para cálculos de média
df_chart['qtde_textos'] = df_va['nota'].value_counts()

nltk.download('stopwords')
nltk.download('punkt') # é um tokenizador, importante para nltk.word_tokenize
# nltk.download('rslp') é um stemmer. acho q nao vou usar
nltk.download('punkt_tab') # Precisei que devia fazer download desse pacote pq o punkt nao tava funcionando

# Pré-processamento streamlit run Tokens_Types.py

#Retirada de sinais gráficos, pontuações e espaços: deixei acentos, retirei números, não fiz stemming
def clean_cp(text):
    cleaned = text.lower() #Deixando tudo minúsculo
    cleaned = re.sub('[^\w\s]', '', cleaned) # Removendo pontuacao
    #cleaned = re.sub('[0-9]+', '', cleaned) # Removendo números 
    cleaned = re.sub('\d+', '', cleaned) # Removendo números
    cleaned = re.sub('\s+', ' ', cleaned) # Removendo espaços extras
    cleaned = re.sub('\s+', ' ', cleaned)
    return cleaned.strip() # Removendo tabs

df_va['content'] = df_va['content'].apply(clean_cp)
#df_va['content'] = df_va['content'].apply(stopwords_cp)

# Tokenizando SEM retirar stopwords: lembrando q token conta duplicados

def tokenized_cp(text):
   #stopwords = nltk.corpus.stopwords.words('portuguese') # Carregando as stopwords do português
   tokenized = nltk.word_tokenize(text, language='portuguese') #Transforma o texto em tokens
   #text_sem_stopwords = [token for token in tokenized if token not in stopwords] # Deixei stopwords pq imaginei que consideravam mas talvez a divergencia de numero q achei em relaçao a
   # dissertacao seja devido às stopwords
   #return text_sem_stopwords
   return tokenized

df_va['tokenized_content'] = df_va['content'].apply(tokenized_cp)

# Verificando a qtde de tokens para cada nota (uso o total pra cada nota pro cálculo de TTR)

# Primeiro conto a qtde de tokens para cada texto e crio uma coluna com a contagem
df_va['token_count'] = df_va['tokenized_content'].apply(len)

# Agora, conto de acordo com cada nota        
df_chart['qtde_total_tokens_nota'] = df_va.groupby('nota')['token_count'].sum()

# Verificando a qtde de tokens MÉDIA para cada nota
df_chart['qtde_media_tokens'] = round((df_chart['qtde_total_tokens_nota'] / df_chart['qtde_textos']),2)

# Separando em types (pego tokens e retiro duplicados)
df_va['types_content'] = df_va['tokenized_content'].apply(lambda tokens: list(set(tokens)))

# Verificando a qtde de types para cada nota

# Primeiro a qtde de types para cada texto e crio uma coluna com a contagem
df_va['types_count'] = df_va['types_content'].apply(len)

# Agora, conto de acordo com cada nota
df_chart['qtde_total_types_nota'] = df_va.groupby('nota')['types_count'].sum()

# Verificando a qtde de types MÉDIA para cada nota ACHO Q NAO VOU USAR
df_chart['qtde_media_types'] = round((df_chart['qtde_total_types_nota'] / df_chart['qtde_textos']),2)

# Cálculo de TTR: TTR = qtde de types / qtde de tokens * 100 (em percentual mesmo)
# Aviso: fiz o cálculo de ttr_medio = (qtde_media_types / qtde_media_tokens) * 100 e obtive os mesmos valores
df_chart['TTR'] = round(((df_chart['qtde_total_types_nota'] / df_chart['qtde_total_tokens_nota']) * 100),2)

# Número mínimo de tokens por nota

# Primeiro conto a qtde de tokens para cada texto e crio uma coluna com a contagem    >> Essas 2 linhas eu repeti mtas vzs no código. É só cortar fora. A variável já foi calculada!
df_va['token_count'] = df_va['tokenized_content'].apply(len)

# Agora, conto o número minimo de tokens para cada nota        
df_chart['qtde_tokens_min'] = df_va.groupby('nota')['token_count'].min()

# Número máximo de tokens por nota

# Primeiro conto a qtde de tokens para cada texto e crio uma coluna com a contagem
df_va['token_count'] = df_va['tokenized_content'].apply(len)

# Agora, conto o número máximo de tokens para cada nota        
df_chart['qtde_tokens_max'] = df_va.groupby('nota')['token_count'].max()

# Desvio padrão do número de tokens por nota

# Primeiro conto a qtde de tokens para cada texto e crio uma coluna com a contagem
df_va['token_count'] = df_va['tokenized_content'].apply(len)

# Agora, conto o número máximo de tokens para cada nota
df_chart['desvpad_qtde_tokens'] = round((df_va.groupby('nota')['token_count'].std()),2)

df_chart['nota_real'] = [3, 2, 4, 5, 1, 0]
df_chart.sort_values('nota_real')

# ==================================================================
# Barra Lateral no Streamlit 
# ==================================================================
st.sidebar.markdown( '## Filtro' )

# Sidebar filter for grade
selected_grades = st.sidebar.multiselect(
    "Escolha a nota que deseja visualizar", [0, 1, 2, 3, 4, 5], default=[0, 1, 2, 3, 4, 5]
)

# Filter DataFrames based on selected grades
filtered_df_chart = df_chart[df_chart['nota_real'].isin(selected_grades)]
filtered_df_va = df_va[df_va['nota'].isin(selected_grades)]

# ==================================================================
# Layout no Streamlit
# ==================================================================
st.header( 'Tokens e Types' )

with st.container():
    st.markdown( """---""" )
    st.subheader( 'Número de Textos' )

    # Create the bar chart
    fig = px.bar(
      filtered_df_chart,  # Pass the entire DataFrame
      x='nota_real',  # X-axis
      y='qtde_textos',  # Y-axis
      labels={'qtde_textos': 'Quantidade', 'nota_real': 'Nota'},  # Custom axis labels
      color_discrete_sequence=['#FF4B4B']  # Streamlit red color
    )

    # Update the tooltip
    fig.update_traces(
       hovertemplate="%{y} textos<extra></extra>"  # Custom tooltip  
    )

    # Update layout for transparent background and tooltip styles
    fig.update_layout(
        #title="Quantidade de Textos por Nota",
        xaxis_title="Nota",
        yaxis_title="Quantidade",
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white"),  # White font for titles, labels, and text
        xaxis=dict(
        showgrid=False,  # Remove vertical gridlines
        color="white"    # White font for x-axis labels
        ),
        yaxis=dict(
        #showgrid=False,  # Remove horizontal gridlines
        color="white"    # White font for y-axis labels
        ),
        hoverlabel=dict(
        bgcolor="#FFFFFF",  # Light red background
        font_size=16,       # Tooltip font size      TROQUEI TAMANHO DA FONTE AQUI
        font_color="black"  # Tooltip text color
        )
    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig )

with st.container():
    st.markdown( """---""" )
    st.subheader( 'Número Máximo e Mínimo de Tokens ')
    
    customdata = filtered_df_chart[['qtde_media_tokens', 'desvpad_qtde_tokens']].values

    # Create the figure
    fig = go.Figure()

    # Add the bars for max tokens
    fig.add_trace(go.Bar(
        x=filtered_df_chart['nota_real'], 
        y=filtered_df_chart['qtde_tokens_max'], 
        name='Máximo de tokens', 
        marker_color='#FF4B4B',  # Streamlit-like red
        hovertemplate=(
            "Máximo de tokens: %{y}<br>" +
            "Número médio: %{customdata[0]:.2f}<br>" +
            "Desvio padrão: %{customdata[1]:.2f}<extra></extra>"     
        ),
        customdata=customdata
    ))

    # Add the bars for min tokens
    fig.add_trace(go.Bar(
        x=filtered_df_chart['nota_real'], 
        y=filtered_df_chart['qtde_tokens_min'], 
        name='Mínimo de tokens', 
        marker_color='#FF904B',  # Nova cor: laranja
        hovertemplate=(
            "Mínimo de tokens: %{y}<br>" +
            "Número médio: %{customdata[0]:.2f}<br>" +
            "Desvio padrão: %{customdata[1]:.2f}<extra></extra>"       
        ),
        customdata=customdata
    ))

    # Update layout for better visualization
    fig.update_layout(
        #title="Números Máximo e Mínimo de Tokens",
        xaxis_title="Nota",
        yaxis_title="Quantidade",
        barmode='group',  # Grouped bars
        legend_title="Legenda",
        template='plotly_white',
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white"),  # White font for titles, labels, and text
        xaxis=dict(
            showgrid=False,  # Remove vertical gridlines
            color="white"    # White font for x-axis labels
        ),
        yaxis=dict(
            #showgrid=False,  # Remove horizontal gridlines
            color="white"    # White font for y-axis labels
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",  # Light red background for tooltip
            font_size=14,       # Tooltip font size        TROQUEI TAMANHO DA FONTE AQUI
            font_color="black"  # Tooltip text color
        )
    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig )

              
with st.container():
    st.markdown( """---""" )
    st.subheader( 'Número Médio de Tokens, Types e TTR' )

    # Create the figure
    fig = go.Figure()

    # Add the bars for avg tokens
    fig.add_trace(go.Bar(
        x=filtered_df_chart['nota_real'], 
        y=filtered_df_chart['qtde_media_tokens'], 
        name='Média de tokens', 
        marker_color='#FF4B4B',  # Vermelho do Streamlit
        hovertemplate=(
            "Média de tokens: %{y}<br>" +
            "TTR: %{customdata:.2f}<extra></extra>%"       
        ),
        customdata=filtered_df_chart['TTR']
    ))

    # Add the bars for avg types
    fig.add_trace(go.Bar(
        x=filtered_df_chart['nota_real'], 
        y=filtered_df_chart['qtde_media_types'], 
        name='Média de types', 
        marker_color='#FF904B',  # Nova cor: laranja
        hovertemplate=(
            "Média de types: %{y}<br>" +
            "TTR: %{customdata:.2f}<extra></extra>%"       
        ),
        customdata=filtered_df_chart['TTR']

    ))

    # Update layout for better visualization
    fig.update_layout(
        #title="Número Médio de Tokens e Types e TTR",
        xaxis_title="Nota",
        yaxis_title="Quantidade",
        barmode='group',  # Grouped bars
        legend_title="Legenda",
        template='plotly_white',
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white"),  # White font for titles, labels, and text
        xaxis=dict(
            showgrid=False,  # Remove vertical gridlines
            color="white"    # White font for x-axis labels
        ),
        yaxis=dict(
            #showgrid=False,  # Remove horizontal gridlines
            color="white"    # White font for y-axis labels
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",  # Light red background for tooltip
            font_size=14,       # Tooltip font size           TROQUEI TAMANHO DA FONTE AQUI
            font_color="black"  # Tooltip text color
        )

    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig )

with st.container():
    st.markdown( """---""" )
    st.subheader( 'Número de Tokens e Types de Cada Texto Segundo Cada Nota' )

    # Create the scatterplot
    fig = px.scatter(
        filtered_df_va,
        x='token_count',                # x-axis: total tokens
        y='types_count',                # y-axis: unique tokens
        color='nota',                   # Color the dots based on the 'nota' column
        hover_data=['token_count', 'types_count', 'nota'],  # Tooltip includes 'nota' #title="Quantidade de Tokens e Types segundo Nota", TA APARECENDO UNDEFINED... COMO TIRAAAAR
        labels={
            'token_count': 'Tokens',
            'types_count': 'Types',
            'nota': 'Nota'
        },
        color_continuous_scale='Agsunset' 
    )

    # Customize layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
        font=dict(color='white'),       # White font for titles, labels, and text
        #title_font=dict(size=18),       # Title font size
        xaxis_title_font=dict(size=16), # X-axis label font size            TROQUEI TAMANHO DA FONTE AQUI
        yaxis_title_font=dict(size=16), # Y-axis label font size               TROQUEI TAMANHO DA FONTE AQUI
    )

    # Customize tooltip
    fig.update_traces(
        hoverlabel=dict(
            bgcolor='white',            # White background for tooltip
            font_size=14,               # Font size for tooltip text        TROQUEI TAMANHO DA FONTE AQUI
            font_color='black'          # Black font color
        )
    )

    # Show the figure
    fig.show()
    st.plotly_chart( fig ) # use_container_width = True pra usar toda largura disponível

