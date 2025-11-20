import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
import numpy as np

# Configurações de Caminho
RATINGS_FILE = os.path.join("data", "usuarios.csv") 
MODEL_FILE = os.path.join("model", "arvore_decisao_model.pkl")

def treinar_modelo_arvore_decisao(df_filmes):
    """
    Treina um modelo de Árvore de Decisão para prever a preferência 
    (gostou/não gostou) com base em duração, gêneros e diretor.
    """
    print("Iniciando treinamento da Árvore de Decisão...")
    try:
        df_ratings = pd.read_csv(RATINGS_FILE)
    except FileNotFoundError:
        print(f"Erro: Arquivo de avaliações '{RATINGS_FILE}' não encontrado. O treinamento não pode prosseguir.")
        return None

    # 1. Criação da Variável Target (Y)
    # Target: Média de rating do filme >= 3.5 (Considerado "Gostou")
    # Garantir que movieId é int para o merge
    df_ratings['movieId'] = pd.to_numeric(df_ratings['movieId'], errors='coerce').astype('Int64')
    media_ratings = df_ratings.groupby('movieId')['rating'].mean().reset_index()
    
    # Converter id_tmdb para int também para match
    df_filmes['id_tmdb'] = pd.to_numeric(df_filmes['id_tmdb'], errors='coerce').astype('Int64')
    
    # Merge com os dados de filmes
    df_modelo = pd.merge(df_filmes, media_ratings, left_on='id_tmdb', right_on='movieId', how='inner')
    df_modelo['gostou'] = (df_modelo['rating'] >= 3.5).astype(int) # 1 = Gostou, 0 = Não Gostou
    
    # 2. Definição de Features (X)
    # Features selecionadas para a Árvore de Decisão
    features_numericas = ['duracao']
    # Usamos 'generos' e 'diretor' como categóricas para influenciar a decisão
    features_categoricas = ['generos', 'diretor'] 
    
    # As colunas categóricas precisam de limpeza para o OneHotEncoder
    for col in features_categoricas:
        df_modelo[col] = df_modelo[col].fillna('').apply(lambda x: str(x).split('|')[0] if '|' in str(x) else str(x))

    df_X = df_modelo[features_numericas + features_categoricas].fillna('')
    y = df_modelo['gostou']

    # 3. Pré-processamento (Pipeline)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_numericas),
            # OneHotEncoder para as features categóricas
            ('cat', OneHotEncoder(handle_unknown='ignore'), features_categoricas)
        ],
        remainder='drop'
    )

    # 4. Criação do Pipeline e Treinamento
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # 5. Salvar o Modelo
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Modelo de Árvore de Decisão treinado e salvo em '{MODEL_FILE}'.")
    return pipeline

def carregar_modelo_arvore_decisao():
    """Carrega o modelo de Árvore de Decisão salvo."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            pipeline = pickle.load(f)
        print("Modelo de Árvore de Decisão carregado.")
        return pipeline
    except FileNotFoundError:
        print(f"ATENÇÃO: Modelo de Árvore de Decisão não encontrado em '{MODEL_FILE}'.")
        return None

def prever_preferencia(pipeline, df_candidatos):
    """Prevê a probabilidade de um filme ser 'gostado' (classe 1)."""
    if pipeline is None:
        return np.full(len(df_candidatos), 0.5) # Retorno neutro se modelo não existir

    # Preparar dados de entrada (Deve corresponder às features usadas no treinamento)
    features_arvore = df_candidatos[['duracao', 'generos', 'diretor']].fillna('')
    
    # Limpeza/Normalização das features categóricas para a predição
    for col in ['generos', 'diretor']:
         features_arvore[col] = features_arvore[col].apply(lambda x: str(x).split('|')[0] if '|' in str(x) else str(x))

    try:
        # Retorna a probabilidade da classe 1 (Gostou)
        probabilidade_gostar = pipeline.predict_proba(features_arvore)[:, 1]
        return probabilidade_gostar
    except Exception as e:
        print(f"Erro ao prever preferência: {e}")
        return np.full(len(df_candidatos), 0.5)
