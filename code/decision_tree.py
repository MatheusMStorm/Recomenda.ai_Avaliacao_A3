import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import numpy as np
from io import StringIO
from sklearn.tree import export_graphviz, plot_tree
try:
    import pydotplus
except Exception:
    pydotplus = None
import matplotlib.pyplot as plt

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
    # Calcular e mostrar eficiência usando a função utilitária
    try:
        from .decision_tree import calcular_eficiencia
    except Exception:
        # se import relativo falhar (execução direta), usar nome do módulo
        try:
            from code.decision_tree import calcular_eficiencia
        except Exception:
            calcular_eficiencia = None

    if callable(calcular_eficiencia):
        try:
            calcular_eficiencia(pipeline, X_test, y_test, verbose=True)
        except Exception as e:
            print(f"Aviso: falha ao calcular eficiência: {e}")
    
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


def calcular_eficiencia(clf, X_test, y_test, verbose: bool = True):
    """Calcula a eficiência (acurácia em %) de um classificador.

    Parâmetros:
    - clf: objeto classificador com método `predict` (ex.: DecisionTreeClassifier ou Pipeline).
    - X_test: dados de teste (features).
    - y_test: rótulos verdadeiros do conjunto de teste.
    - verbose: se True, imprime a acurácia e relatório de classificação.

    Retorna:
    - acuracia_percent: acurácia em porcentagem (float).
    - y_pred: array com as previsões feitas pelo classificador.
    """
    if clf is None:
        raise ValueError("O classificador (clf) não pode ser None")

    # Obter previsões
    y_pred = clf.predict(X_test)

    # Calcular acurácia
    acuracia_percent = accuracy_score(y_test, y_pred) * 100.0

    if verbose:
        print(f"Eficiência do modelo (Acurácia): {acuracia_percent:.2f}%")
        try:
            print("\nClassification report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        except Exception:
            pass

    return acuracia_percent, y_pred


def plotar_arvore_decisao(pipeline_or_clf, X_sample=None, output_file=None, figsize=(20, 10), class_names=None):
    """Plota a árvore de decisão.

    - Se `pipeline_or_clf` for um `Pipeline`, a função extrai o estimador final (`classifier`) e
      tenta recuperar os nomes das features a partir do `preprocessor` usando `X_sample`.
    - Tenta usar Graphviz/pydotplus para gerar PNG; se não disponível, usa `matplotlib`.

    Parâmetros:
    - pipeline_or_clf: Pipeline ou classificador (DecisionTreeClassifier)
    - X_sample: DataFrame com features brutas (necessário para recuperar nomes após transformação)
    - output_file: caminho do arquivo de saída (ex.: 'model/arvore_decisao.png'). Se None, usa
      'model/arvore_decisao.png' por padrão.
    - figsize: tupla para o tamanho da figura (quando usa matplotlib)
    - class_names: lista opcional com nomes das classes (ex.: ['Não Gostou','Gostou'])

    Retorna o caminho do arquivo gravado ou None se falhar.
    """
    if output_file is None:
        output_file = os.path.join('model', 'arvore_decisao.png')

    # Extrair estimator e tentar obter feature names
    clf = pipeline_or_clf
    feature_names = None
    if hasattr(pipeline_or_clf, 'named_steps'):
        # Pipeline
        try:
            preprocessor = pipeline_or_clf.named_steps.get('preprocessor')
            clf = pipeline_or_clf.named_steps.get('classifier', pipeline_or_clf)
            if preprocessor is not None and X_sample is not None:
                try:
                    # scikit-learn >=1.0 supports get_feature_names_out
                    feature_names = preprocessor.get_feature_names_out()
                except Exception:
                    # construir manualmente a partir dos transformers_
                    names = []
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'remainder' and transformer == 'drop':
                            continue
                        if hasattr(transformer, 'get_feature_names_out') and cols is not None:
                            try:
                                out = transformer.get_feature_names_out(cols)
                                names.extend(list(out))
                                continue
                            except Exception:
                                pass
                        # caso seja StandardScaler ou similar
                        if cols is not None:
                            if hasattr(transformer, 'categories_'):
                                # OneHotEncoder
                                cats = transformer.categories_
                                for i, col in enumerate(cols):
                                    for cat in cats[i]:
                                        names.append(f"{col}_{cat}")
                            else:
                                names.extend(list(cols))
                    feature_names = names
        except Exception:
            feature_names = None

    # Se não foi possível inferir feature_names mas X_sample fornecido, usar colunas
    if feature_names is None and X_sample is not None:
        try:
            feature_names = list(X_sample.columns)
        except Exception:
            feature_names = None

    # Definir nomes de classes
    if class_names is None:
        class_names = ['0', '1']

    # Primeiro caminho: tentar Graphviz via pydotplus
    try:
        if pydotplus is not None:
            dot_data = StringIO()
            export_graphviz(clf, out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True,
                            feature_names=feature_names,
                            class_names=class_names)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            graph.write_png(output_file)
            print(f"Árvore de decisão salva em: {output_file} (via pydotplus)")
            return output_file

    except Exception as e:
        print(f"Aviso: falha ao gerar imagem via pydotplus/graphviz: {e}")

    # Fallback: usar matplotlib
    try:
        plt.figure(figsize=figsize)
        plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Árvore de decisão salva em: {output_file} (via matplotlib)")
        return output_file
    except Exception as e:
        print(f"Erro ao plotar árvore de decisão: {e}")
        return None
