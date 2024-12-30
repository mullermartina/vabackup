import pandas as pd
import numpy as np
import nltk
from collections import Counter #uso para n-gramas
import re # talvez regex vai ajudar na normalizaçao
import plotly.graph_objects as go
import streamlit as st

# ==================================================================
# Configurações da Página 
# ==================================================================
st.set_page_config(page_title = 'Frequência das Palavras',
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
df_va.groupby('nota').count() # ACXHO Q NAO PRECISO DESSA LINHA MAS NAO CHEGUEI A TESTAR SEM ELA !!

nltk.download('stopwords')
nltk.download('punkt') # é um tokenizador, importante para nltk.word_tokenize
nltk.download('punkt_tab') # download desse pacote pq o punkt sozinho nao tava funcionando

# Pré-processamento

#Retirada de sinais gráficos, pontuações e espaços
def clean_cp(text):
    cleaned = text.lower() #Deixando tudo minúsculo
    cleaned = re.sub('[^\w\s]', '', cleaned) # Removendo pontuacao
    #cleaned = re.sub('[0-9]+', '', cleaned) # Removendo números 
    cleaned = re.sub('\d+', '', cleaned) # Removendo números
    cleaned = re.sub('\s+', ' ', cleaned) # Removendo espaços extras
    cleaned = re.sub('\s+', ' ', cleaned)
    return cleaned.strip() # Removendo tabs

df_va['content'] = df_va['content'].apply(clean_cp)

# Tokenizando e retirando stopwords: importante para ver a frequencia das palavras

def tokenized_cp(text):
   stopwords = nltk.corpus.stopwords.words('portuguese') # Carregando as stopwords do português
   tokenized = nltk.word_tokenize(text, language='portuguese') #Transforma o texto em tokens
   text_sem_stopwords = [token for token in tokenized if token not in stopwords] # Deixando somente o que nao é stopword no texto
   return text_sem_stopwords

df_va['tokenized_content'] = df_va['content'].apply(tokenized_cp)


# ==================================================================
# Processamentos específicos para essa visualização
# ==================================================================

# Contando o número de tokens SEM STOPWORDS 

# Contando tokens sem considerar a nota
df_va['num_tokens'] = df_va['tokenized_content'].apply(len)

# Agrupando de acordo com cada nota
qtde_tokens_nota = df_va.groupby('nota')['num_tokens'].sum()

# Verificando a ocorrência dos tokens (!!!!!!!! não ta no frequenciapalavrasv2_streamlit)

# Criando uma lista só com todos os tokens
#all_tokens = [token for tokens in df_va['tokenized_content'] for token in tokens]

# Contando a ocorrência de cada token
#token_counts = Counter(all_tokens) até aaqui não tá no arquivo acima

# Agrupando a contagem de tokens de acordo com cada nota
nota_token_counts = (
    df_va.groupby('nota')['tokenized_content']
    .apply(lambda texts: Counter([token for text in texts for token in text]))
)

# Convertendo cada token e contagem em um dataframe
df_frequency = nota_token_counts.reset_index()
df_frequency.columns = ['nota', 'token', 'token_frequency']
sorted_tokens = df_frequency.sort_values('token_frequency', ascending=False)

df_words = pd.DataFrame()

# Agrupando a contagem de tokens por nota
top_tokens_per_grade = (
    sorted_tokens.groupby('nota')
    .head(100)  # Assim mostra os 100 tokens mais comuns. Já vai ficar péssimo na visualização entao mantenho esse head e insiro outro abaixo.
    .reset_index(drop=True)
)
df_words = top_tokens_per_grade

#df_words_sorted = (df_words.sort_values(by=['nota', 'token_frequency'], ascending=[True, False]))
# Testei e nãão é necessário. De qlqr forma, tá referenciado.. Só mudar no gráfico abaixo

# ==================================================================
# Processamentos específicos para o segundo heatmap
# ==================================================================
file_path = 'texto_tarefa4.txt'

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    given_text = file.read()

# Pré-processamento do texto de apoio + texto da tarefa
cleaned_text = clean_cp(given_text)
tokenized_text = tokenized_cp(cleaned_text)

# Retirando tokens do texto de apoio do conjunto de tokens dos textos dos candidatos
# Ensure tokenized_text is a set for efficient subtraction
tokenized_text_set = set(tokenized_text) # Using a set for tokenized_text ensures efficient lookups when checking if a token should be excluded.

# Subtract tokenized_text from each row in 'tokenized_content'
df_va['filtered_content'] = df_va['tokenized_content'].apply(
    lambda tokens: [token for token in tokens if token not in tokenized_text_set]
)

# Contando o número de tokens SEM STOPWORDS (diferente do typestokenttr.ipynb) para cada texto: usarei isso para o gráfico 2, que demonstra o número mínimo e número máximo de tokens
# Contando tokens sem considerar a nota
df_va['num_tokens'] = df_va['filtered_content'].apply(len)

# Agrupando de acordo com cada nota
qtde_tokens_nota = df_va.groupby('nota')['num_tokens'].sum()

# Agrupando a contagem de tokens de acordo com cada nota MANTER
nota_token_counts = (
    df_va.groupby('nota')['filtered_content']
    .apply(lambda texts: Counter([token for text in texts for token in text]))
)

# Convertendo cada token e contagem em um dataframe
df_frequency = nota_token_counts.reset_index()
df_frequency.columns = ['nota', 'token', 'token_frequency']
sorted_tokens = df_frequency.sort_values('token_frequency', ascending=False)

df_words_ta = pd.DataFrame()

# Agrupando a contagem de tokens por nota de forma a mostrar os 15 mais comuns para cada nota ( 15 * 6 = 90 linhas portanto )
top_tokens_per_grade = (
    sorted_tokens.groupby('nota')
    .head(100)  # Assim mostra os 1000 tokens mais comuns. Já vai ficar péssimo na visualização entao mantenho esse head e insiro outro abaixo.
    .reset_index(drop=True)
)
df_words_ta = top_tokens_per_grade

# ==================================================================
# Barra Lateral no Streamlit 
# ==================================================================
st.sidebar.markdown( '## Filtros' )

# Sidebar filter for grade
selected_grades = st.sidebar.multiselect(
    "Escolha a nota que deseja visualizar", [0, 1, 2, 3, 4, 5], default=[0, 1, 2, 3, 4, 5]
)

# Filter Dataframes based on selected grades
filtered_df_words = df_words[df_words['nota'].isin(selected_grades)]

st.markdown("")

# Filtro para escolher um número x de palavras mais frequentes
number = st.sidebar.slider(
    "Escolha o número de palavras mais frequentes que você deseja exibir",
    min_value=1,  # Minimum value of the slider
    max_value=50,  # Maximum value of the slider
    value=15,  # Default value
    step=1  # Step size
)

# ==================================================================
# Layout no Streamlit
# ==================================================================
st.header( 'Frequência das palavras' )

with st.container():
    st.markdown( """---""" )
    st.subheader(f'{number} Palavras mais frequentes')

    # Pivot the data for heatmap
    heatmap_data = filtered_df_words.pivot(index='token', columns='nota', values='token_frequency').fillna(0).head(number)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,  # Frequency matrix
        x=heatmap_data.columns,  # Grades (x-axis)
        y=heatmap_data.index,    # Tokens (y-axis)
        colorscale='Reds',       # Heatmap color scale
        colorbar=dict(title="Frequência", titlefont=dict(size=14), tickfont=dict(size=12)),  # Colorbar title
        hoverongaps=False,       # Ensure no gaps show on hover
        hovertemplate=(
            "<span style='font-size:14px'><b>Token</b>: %{y}<br>" +  # Tooltip with larger font
            "<b>Nota</b>: %{x}<br>" +
            "<b>Frequência</b>: %{z}</span><extra></extra>"
        )
    ))

    # Update layout for better visualization
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white", size=14),  # White font for all text
        height=600, # Increase height for better visualization
        width=1000
    )


    #fig.update_yaxes(title="Tokens", tickfont=dict(size=14), automargin=True)
    fig.update_yaxes(title="Tokens", showticklabels=False) # showticklabels=False para nao mostrar as palavras à esquerda, somente no tooltip
    fig.update_xaxes(title="", showticklabels=False)

    # Show the figure
    fig.show()
    st.plotly_chart( fig, use_container_width = True )

with st.container():
    st.subheader(f'{number} Palavras mais frequentes que não estão no enunciado ou no texto de apoio')

    # Pivot the data for heatmap
    heatmap_data = df_words_ta.pivot(index='token', columns='nota', values='token_frequency').fillna(0).head(number)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,  # Frequency matrix
        x=heatmap_data.columns,  # Grades (x-axis)
        y=heatmap_data.index,    # Tokens (y-axis)
        colorscale='Reds',       # Heatmap color scale
        colorbar=dict(title="Frequência", titlefont=dict(size=14), tickfont=dict(size=12)),  # Legend title and font size
        hoverongaps=False,       # Ensure no gaps show on hover
        hovertemplate=(
            "<span style='font-size:14px'><b>Token</b>: %{y}<br>" +  # Tooltip with larger font
            "<b>Nota</b>: %{x}<br>" +
            "<b>Frequência</b>: %{z}</span><extra></extra>"
        )
    ))

    # Update layout for better visualization
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent outer background
        font=dict(color="white", size=14),  # Update general font size
        height=600,  # Adjust height
        width=1000,   # Keep width consistent
    )

    # Update y-axis to hide tokens
    fig.update_yaxes(
        title="",  # No y-axis title
        showticklabels=False,  # Hide tokens from y-axis
    )

    # Update x-axis for better styling
    fig.update_xaxes(title="", showticklabels=False)

    # Show the figure
    fig.show()
    st.plotly_chart( fig, use_container_width = True )