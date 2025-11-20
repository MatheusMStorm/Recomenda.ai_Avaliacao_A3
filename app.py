import streamlit as st
import pandas as pd
from code import recomendar
import os

# --- Configuração Inicial ---
st.set_page_config(page_title="Recomenda.ai - Sistema de Recomendação Híbrida")
st.image("img/Recomenda.ai.png", width=100)
st.title("🎬 Recomenda.ai")

def _format_filter_summary(parsed_filters):
    """Format parsed filters into a readable summary for display."""
    parts = []
    
    if parsed_filters['diretores']:
        dirs = ", ".join([f"'{d}'" for d in parsed_filters['diretores']])
        parts.append(f"**Diretores**: {dirs}")
    
    if parsed_filters['atores']:
        ators = ", ".join([f"'{a}'" for a in parsed_filters['atores']])
        parts.append(f"**Atores**: {ators}")
    
    if parsed_filters['generos']:
        gens = ", ".join([f"'{g}'" for g in parsed_filters['generos']])
        parts.append(f"**Gêneros**: {gens}")
    
    if parsed_filters['text_tokens']:
        tokens = ", ".join([f"'{t}'" for t in parsed_filters['text_tokens']])
        parts.append(f"**Busca Livre**: {tokens}")
    
    if parsed_filters['dur_min'] is not None:
        parts.append(f"**Duração Mínima**: {parsed_filters['dur_min']} min")
    
    if parsed_filters['dur_max'] is not None:
        parts.append(f"**Duração Máxima**: {parsed_filters['dur_max']} min")
    
    return " | ".join(parts) if parts else None

def main():
    st.markdown("Este sistema utiliza **PNL (Similaridade de Conteúdo)** para gerar candidatos e uma **Árvore de Decisão** para filtrar e pontuar as recomendações com base em Duração e Gêneros.")

    # Tenta carregar os modelos (lazy loader) e informa o usuário se o treinamento é necessário
    try:
        # `recomendar` fornece uma função para carregar os modelos sob demanda
        if hasattr(recomendar, "_load_models"):
            recomendar._load_models()
    except Exception as e:
        st.warning(f"Problema ao carregar modelos: {e}")

    if recomendar.DF_FILMES is None or recomendar.MODELO_ARVORE is None:
        st.error("Modelos de IA não encontrados. Por favor, execute o script `main.py` no terminal para treinar os modelos primeiro.")
        return

    st.subheader("Filtragem e Recomendação")
    
    # Escolher modo de busca
    modo = st.radio(
        "Como você deseja descobrir filmes?",
        options=["🎯 Por um Filme Similar", "🔍 Por Filtros (Diretor, Ator, Gênero, etc)"],
        index=0
    )
    
    num_recomenda = st.slider(
        "Número de Recomendações:",
        min_value=5, max_value=20, value=10
    )

    if modo == "🎯 Por um Filme Similar":
        # Modo 1: Baseado em um título
        titulo_base = st.text_input(
            "Digite o Título do Filme Base (para encontrar similares):",
            value="Matrix"
        )
        filtros_texto = st.text_area(
            "Refinar com filtros opcionais:",
            value="",
            help="Opcional: Tom Hanks, Ação, duracao: 90-120, etc. Ou deixe em branco."
        )
        
        if st.button("Recomendar"):
            if titulo_base:
                with st.spinner(f"Processando recomendações para '{titulo_base}'..."):
                    resultados = recomendar.recomendar_filmes_hibrido(titulo_base, num_recomenda, filtros_texto=filtros_texto)

                    if isinstance(resultados, str):
                        st.warning(resultados)
                    else:
                        # Parse e exibe filtros aplicados
                        parsed = recomendar._parse_filters(filtros_texto) if filtros_texto else {}
                        filter_summary = _format_filter_summary(parsed) if parsed else None
                        
                        if filter_summary:
                            st.markdown(f"**🔍 Filtros Aplicados:** {filter_summary}")
                        
                        df_resultados = pd.DataFrame(resultados)
                        st.success("Recomendações Encontradas:")
                        
                        # Estiliza o DataFrame
                        st.dataframe(
                            df_resultados.style.format({'Score Híbrido': "{:.4f}"}),
                            hide_index=True
                        )
                        
                        st.info("O **Score Híbrido** é a multiplicação da Similaridade do PNL pela Probabilidade de Preferência da Árvore de Decisão.")
            else:
                st.warning("Por favor, digite um título para iniciar a recomendação.")
    
    else:
        # Modo 2: Apenas por filtros
        st.info("💡 **Dica:** Você pode digitar de forma livre! O sistema reconhece automaticamente diretores, atores, gêneros e duração. Exemplos:\n"
                "- `Tom Hanks` (busca por ator)\n"
                "- `Scorsese Drama` (diretor + gênero)\n"
                "- `Acao Animacao duracao: 90-120` (gêneros + duração)\n"
                "- `Christopher Nolan, Ficção científica, >150` (misturado com vírgulas ou sem)")
        
        filtros_texto = st.text_area(
            "Digite seus interesses (não precisa de formatação específica):",
            value="",
            help="Escreva como quiser! Tom Hanks, Ação, Drama, Scorsese, duracao: 100-180, etc."
        )
        
        if st.button("Descobrir Filmes"):
            if filtros_texto.strip():
                with st.spinner("Buscando filmes com seus critérios..."):
                    resultados = recomendar.descobrir_filmes_por_filtros(filtros_texto, num_recomenda)

                    if isinstance(resultados, str):
                        st.warning(resultados)
                    else:
                        # Parse e exibe filtros aplicados
                        parsed = recomendar._parse_filters(filtros_texto)
                        filter_summary = _format_filter_summary(parsed) if parsed else None
                        
                        if filter_summary:
                            st.markdown(f"**🔍 Critérios Aplicados:** {filter_summary}")
                        
                        df_resultados = pd.DataFrame(resultados)
                        st.success("Filmes Encontrados (Ordenados por Preferência Estimada):")
                        
                        # Estiliza o DataFrame
                        st.dataframe(
                            df_resultados.style.format({'Score de Preferência': "{:.4f}"}),
                            hide_index=True
                        )
                        
                        st.info("O **Score de Preferência** é a probabilidade da Árvore de Decisão de que você goste do filme.")
            else:
                st.warning("Por favor, forneça pelo menos um critério de filtro.")

if __name__ == "__main__":
    # Garante que os caminhos de importação estão corretos
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
    
    main()