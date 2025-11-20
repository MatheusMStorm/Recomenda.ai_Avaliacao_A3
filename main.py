import pandas as pd
from code import coleta_api, pnl_module, decision_tree, recomendar

def treinar_todos_modelos(df_filmes):
    """Orquestra o treinamento do PNL e da Árvore de Decisão."""
    if df_filmes.empty:
        print("Não há dados para treinar. Verifique 'Data/filmes.csv'.")
        return

    # 1. Treinamento do Módulo PNL (Cria a Matriz de Similaridade)
    print("\n--- Treinando Modelo PNL ---")
    pnl_module.treinar_modelo_pnl(df_filmes.copy())

    # 2. Treinamento do Módulo Árvore de Decisão (Cria o Pipeline de Classificação)
    print("\n--- Treinando Modelo Árvore de Decisão ---")
    decision_tree.treinar_modelo_arvore_decisao(df_filmes.copy())
    
    print("\n--- Treinamento Concluído com Sucesso ---")

def executar_recomendacao(titulo):
    """Testa a recomendação no terminal."""
    print(f"\n--- Recomendando Filmes Similares a '{titulo}' ---")
    
    # A função 'recomendar_filmes_hibrido' carrega os modelos automaticamente
    resultados = recomendar.recomendar_filmes_hibrido(titulo)
    
    if isinstance(resultados, str):
        print(resultados)
    else:
        df_resultado = pd.DataFrame(resultados)
        print(df_resultado.to_markdown(index=False))

if __name__ == "__main__":
    df_dados = coleta_api.carregar_e_limpar_dados()
    
    # Opção 1: Treinar os modelos
    treinar_todos_modelos(df_dados)
    
    # Opção 2: Executar a recomendação (pressupõe que os modelos foram treinados)
    titulo_teste = "Toy Story" # Exemplo de filme base
    executar_recomendacao(titulo_teste) 
    
    # Para iniciar a interface Streamlit, execute no terminal:
    # streamlit run app.py