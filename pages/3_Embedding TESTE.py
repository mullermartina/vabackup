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
    st.markdown("""---""")
    st.subheader('Embedding das Palavras do Corpus')

    # TSNE visualization function
    def visualize_embeddings(model, selected_word=None, num_neighbors=10, num_points=1000):
        words = list(model.wv.index_to_key)[:num_points]  # Limit the number of words to display
        vectors = model.wv[words]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_vectors = tsne.fit_transform(vectors)

        df_fe = pd.DataFrame({
            'Token': words,
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1]
        })

        if selected_word:
            # Find the neighbors of the selected word
            neighbors = model.wv.most_similar(selected_word, topn=num_neighbors)
            neighbor_words = [selected_word] + [word for word, _ in neighbors]
            df_filtered = df_fe[df_fe['Token'].isin(neighbor_words)]
            highlight_color = '#FFA500'  # Highlight color for selected word
            df_fe['color'] = df_fe['Token'].apply(
                lambda x: highlight_color if x == selected_word else '#FF4B4B'
            )
            df_filtered['color'] = df_filtered['Token'].apply(
                lambda x: highlight_color if x == selected_word else '#FF4B4B'
            )
        else:
            df_filtered = df_fe
            df_fe['color'] = '#FF4B4B'

        fig = px.scatter(
            df_filtered, x='x', y='y', text='Token',
            labels={'x': 'Dimensão 1', 'y': 'Dimensão 2'},
            color=df_filtered['color']
        )
        fig.update_traces(
            textposition='top center',
            marker=dict(size=10),  # Adjust marker size
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

    # Streamlit UI
    visualization_choice = st.radio(
        "Você quer visualizar todas as palavras ou apenas uma palavra específica e seus vizinhos?",
        options=["Visualizar todas as palavras", "Visualizar uma palavra específica e seus vizinhos"]
    )

    if visualization_choice == "Visualizar todas as palavras":
        st.write("Visualizando todas as palavras.")
        fig = visualize_embeddings(model)
    else:
        word_list = list(model.wv.index_to_key)  # Full list of vocabulary
        selected_word = st.selectbox('Escolha uma palavra para visualizar:', word_list)
        if selected_word:
            st.write(f'Visualizando a palavra **{selected_word}** e seus 10 vizinhos mais próximos.')
            fig = visualize_embeddings(model, selected_word)
        else:
            st.write("Por favor, escolha uma palavra para visualizar.")

    st.plotly_chart(fig, use_container_width=True)