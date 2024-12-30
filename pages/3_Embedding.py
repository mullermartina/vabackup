import pandas as pd
import numpy as np
import nltk
from collections import Counter #uso para n-gramas
import re 
import plotly.express as px #biblio q pus pra essa viz
from sklearn.manifold import TSNE #biblio q pus pra essa viz
from gensim.models import Word2Vec
import streamlit as st

# ==================================================================
# Configurações da Página 
# ==================================================================
st.set_page_config(page_title = 'Embedding Fixa',
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

nltk.download('stopwords')
nltk.download('punkt') # é um tokenizador, importante para nltk.word_tokenize
nltk.download('punkt_tab') # download desse pacote pq o punkt sozinho nao tava funcionando

# Pré-processamento
#Retirada de sinais gráficos, pontuações e espaços
def clean_cp(text):
    cleaned = text.lower() #Deixando tudo minúsculo
    cleaned = re.sub('[^\w\s]', '', cleaned) # Removendo pontuacao
    #cleaned = re.sub('[0-9]+', '', cleaned) # Removendo números 
    cleaned = re.sub('\d+', '', cleaned) # Removendo números NÃO TAVA FUNCIONANDO. Começou a funcionar qdo pus o lower como primeiro comando da funçao
    cleaned = re.sub('\s+', ' ', cleaned) # Removendo espaços extras
    cleaned = re.sub('\s+', ' ', cleaned)
    return cleaned.strip() # Removendo tabs

df_va['content'] = df_va['content'].apply(clean_cp)

# Tokenizando e retirando stopwords: retiro stopwords pq embeddings fixas nao consideram contexto
def tokenized_cp(text):
   stopwords = nltk.corpus.stopwords.words('portuguese') # Carregando as stopwords do português
   tokenized = nltk.word_tokenize(text, language='portuguese') #Transforma o texto em tokens
   text_sem_stopwords = [token for token in tokenized if token not in stopwords] # Deixando somente o que nao é stopword no texto
   return text_sem_stopwords

df_va['tokenized_content'] = df_va['content'].apply(tokenized_cp)

# AQUI Q VAI ALGUM FILTRO? OU METER DIRETO NA PÁGINA?

# Prepare the tokenized corpus
tokenized_corpus = df_va['tokenized_content'].tolist()

# ==================================================================
# Treinamento da Embedding
# ==================================================================
model = Word2Vec( # Se ficar pesado, inserir no streamlit somente o modelo treinado já
    sentences=tokenized_corpus,  # Input tokenized sentences
    vector_size=100,            # Dimensionality of the embedding vectors
    window=5,                   # Context window size
    min_count=1,                # Minimum word frequency
    sg=0,                       # CBOW (0) or Skip-gram (1)
    workers=4                   # Number of worker threads
)


# ==================================================================
# Layout no Streamlit
# ==================================================================
st.header( 'Embedding Fixa ' )

with st.container():
    st.markdown( """---""" )
    st.subheader( 'Embedding das Palavras do Corpus' )


    # TSNE visualization function

    def visualize_embeddings(model, num_points=1000):
        words = list(model.wv.index_to_key)[:num_points]  # num_points limits the number of words to display
        vectors = model.wv[words]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_vectors = tsne.fit_transform(vectors)

        df_fe = pd.DataFrame({
            'Token': words,
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1]
        })

        fig = px.scatter(
            df_fe, x='x', y='y', text='Token',
            #title="t-SNE Visualization of Word Embeddings",
            labels={'x': 'Dimensão 1', 'y': 'Dimensão 2'}
        )
        fig.update_traces(
            textposition='top center',
            marker=dict(color='#FF4B4B'),  # Streamlit red color
            textfont=dict(color='white')  # White words
        )
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent outer background
            font=dict(color='white'),  # White font for axis labels and title
            hoverlabel=dict(
                bgcolor='white',  # White background for tooltips
                font_size=12,     # Tooltip font size
                font_color='black'  # Tooltip font color
            )
        )
        return fig

    visualize_embeddings(model)
    st.plotly_chart( visualize_embeddings(model), use_container_width=True )
