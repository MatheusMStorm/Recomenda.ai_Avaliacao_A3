# Codigo_fonte/coleta_api.py (Adaptado)

import pandas as pd
import os

DATA_PATH = os.path.join("data", "filmes.csv")

def carregar_e_limpar_dados():
    """Carrega o arquivo CSV e garante que os campos estejam prontos."""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # 1. Garantir as colunas necessárias para PNL e Árvore
        required_cols = ['titulo', 'diretor', 'atores', 'sinopse', 'generos', 'duracao']
        
        # Se movieId existe, renomear para id_tmdb, caso contrário criar
        if 'movieId' in df.columns:
            df['id_tmdb'] = df['movieId']
        else:
            print(f"Alerta: Coluna 'movieId' faltando. Adicionando com valores vazios.")
            df['id_tmdb'] = ''
        
        for col in required_cols:
            if col not in df.columns:
                print(f"Alerta: Coluna '{col}' faltando. Adicionando com valores vazios.")
                df[col] = ''
                
        # 2. Conversão de Duração (Essencial para a Árvore de Decisão)
        df['duracao'] = pd.to_numeric(df['duracao'], errors='coerce').fillna(0) 
        df['duracao'] = df['duracao'].astype(int)
        
        print(f"Dados carregados do {DATA_PATH} com {len(df)} filmes.")
        return df[['id_tmdb'] + required_cols]

    except FileNotFoundError:
        print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
        # Em um sistema real, aqui você chamaria a API para criar o arquivo
        return pd.DataFrame() 

# Funções de busca e coleta da API (mantidas fora do escopo para focar na IA, mas devem existir aqui)
# Exemplo: def coletar_dados_da_api_e_salvar(api_key): ...