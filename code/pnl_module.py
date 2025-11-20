import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import pickle

DATA_PATH = os.path.join('data', 'filmes.csv')
MODEL_PATH = os.path.join('model', 'pnl_similarity.pkl')

def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', str(texto)).lower()
    return texto

def processar_lista(lista_str):
    if pd.isna(lista_str):
        return ""
    
    return ' '.join([''.join(i.split(' ')) for i in str(lista_str).split('|')]).lower()

def criar_features_de_texto(df):
    print("Criando features de texto...")

    df['generos'] = df['generos'].fillna('')
    df['diretor'] = df['diretor'].fillna('')
    df['atores'] = df['atores'].fillna('')
    df['titulo'] = df['titulo'].fillna('')

    df['generos_limpo'] = df['generos'].apply(processar_lista)
    df['diretor_limpo'] = df['diretor'].apply(processar_lista)
    df['atores_limpo'] = df['atores'].apply(processar_lista)

    df['feature_pnl'] = (
        df['titulo'].apply(lambda x: str(x).lower().replace(' ', '')) + ' ' +
        df['generos_limpo'] + ' ' +
        df['diretor_limpo'] + ' ' +
        df['atores_limpo']
    )

    return df

def treinar_modelo_pnl(df):
    print("Treinando modelo PNL...")
    df = criar_features_de_texto(df.copy())

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['feature_pnl'])

    print("Calculando similaridade de cosseno...")

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print("Salvando modelo treinado...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((cosine_sim, df.reset_index()), f)

    print("Modelo salvo em:", MODEL_PATH)
    return cosine_sim, df.reset_index()

def carregar_modelo_pnl():
    try:
        with open(MODEL_PATH, 'rb') as f:
            cosine_sim, df = pickle.load(f)
        print("Modelo carregado com sucesso.")
        return cosine_sim, df
    except FileNotFoundError:
        print(f"ATENÇÃO: Modelo PNL não encontrado em '{MODEL_PATH}'. Treine o modelo primeiro.")
        return None, None

def obter_similares(df_filmes, cosine_sim, titulo_base, top_n=50):
    idx = df_filmes[df_filmes['titulo'].str.contains(titulo_base, case=False, na=False)].index
    if idx.empty:
        print(f"Filme com título contendo '{titulo_base}' não encontrado.")
        return pd.DataFrame()
    
    idx = idx[0] # Pega o primeiro índice correspondente

    # Obter as pontuações de similaridade
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar as pontuações de similaridade
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores] # Índices dos filmes similares
    similaridade = [i[1] for i in sim_scores] # Pontuações de similaridade

    candidatos_df = df_filmes.iloc[movie_indices].copy()
    candidatos_df['similaridade'] = similaridade

    return candidatos_df

def carregar_dados():
    try:
        df = pd.read_csv(DATA_PATH)
        # Garantir que a coluna 'duracao' seja numérica
        df['duracao'] = pd.to_numeric(df['duracao'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        print(f"ATENÇÃO: Arquivo de dados não encontrado em '{DATA_PATH}'.")
        return None