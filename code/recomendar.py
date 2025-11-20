import pandas as pd
import re
import unicodedata

from code import pnl_module
from code import decision_tree

def _normalize(text):
    """Remove accents and convert to lowercase for robust matching."""
    if not isinstance(text, str):
        return str(text).lower()
    # Remove accents using NFD decomposition
    nfd = unicodedata.normalize('NFD', text)
    cleaned = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return cleaned.lower()

# Carrega os modelos globais uma única vez
COSINE_SIM, DF_FILMES = None, None
MODELO_ARVORE = None

def _load_models():
    """Carrega os modelos de forma lazy."""
    global COSINE_SIM, DF_FILMES, MODELO_ARVORE
    if COSINE_SIM is None:
        COSINE_SIM, DF_FILMES = pnl_module.carregar_modelo_pnl()
    if MODELO_ARVORE is None:
        MODELO_ARVORE = decision_tree.carregar_modelo_arvore_decisao()


def _smart_parse_filters(filter_text, df_filmes=None):
    """
    Parse filters com inteligência heurística.
    Não requer prefixos (diretor:, ator:, etc).
    Tenta reconhecer automaticamente o tipo de cada token.
    
    Se df_filmes é fornecido, faz matching inteligente contra atores/diretores conhecidos.
    """
    out = {
        'diretores': [],
        'atores': [],
        'generos': [],
        'text_tokens': [],
        'dur_min': None,
        'dur_max': None,
    }

    if not filter_text or not isinstance(filter_text, str):
        return out

    # Gêneros conhecidos (lista comum)
    generos_conhecidos = [
        'acao', 'adventure', 'animacao', 'animation', 'comedy', 'comedia', 'drama',
        'fantasia', 'fantasy', 'ficção científica', 'scifi', 'sci-fi', 'horror',
        'terror', 'romance', 'thriller', 'mystery', 'mistério', 'crime', 'western',
        'documentário', 'documentary', 'musical', 'aventura', 'familia', 'family'
    ]

    # Split por separadores comuns (vírgula, ponto-e-vírgula, quebra de linha)
    # Se não houver separadores, tenta quebrar por espaços também
    if any(sep in filter_text for sep in [',', ';', '\n']):
        parts = [p.strip() for p in re.split(r'[;,\n]+', filter_text) if p.strip()]
    else:
        # Sem separadores explícitos: quebra por espaços, mantendo palavras de 2+ caracteres
        tokens = filter_text.split()
        parts = [t for t in tokens if len(t) > 1]

    for part in parts:
        low = part.lower()

        # 1. Tenta reconhecer prefixos explícitos (compatibilidade com parser antigo)
        if low.startswith('diretor:') or low.startswith('director:'):
            name = part.split(':', 1)[1].strip()
            if name:
                out['diretores'].append(name)
            continue

        if low.startswith('ator:') or low.startswith('atores:'):
            names = part.split(':', 1)[1].strip()
            for n in re.split(r'[,/]+', names):
                n = n.strip()
                if n:
                    out['atores'].append(n)
            continue

        if low.startswith('genero:') or low.startswith('generos:') or low.startswith('gênero:') or low.startswith('gêneros:'):
            if ':' in part:
                vals = part.split(':', 1)[1]
                for g in re.split(r'[,/]+', vals):
                    g = g.strip()
                    if g:
                        out['generos'].append(g)
                continue

        # 2. Tenta reconhecer duração (padrões: 90-120, >=90, >90, <=120, <120)
        if 'durac' in low or 'duraç' in low:
            if '-' in low:
                m = re.search(r'(\d{1,4})\s*-\s*(\d{1,4})', low)
                if m:
                    out['dur_min'] = int(m.group(1))
                    out['dur_max'] = int(m.group(2))
                    continue
            # relational ops
            m_ge = re.search(r'>=\s*(\d{1,4})', low)
            m_gt = re.search(r'>\s*(\d{1,4})', low)
            m_le = re.search(r'<=\s*(\d{1,4})', low)
            m_lt = re.search(r'<\s*(\d{1,4})', low)
            if m_ge:
                out['dur_min'] = int(m_ge.group(1))
                continue
            if m_gt:
                out['dur_min'] = int(m_gt.group(1)) + 1
                continue
            if m_le:
                out['dur_max'] = int(m_le.group(1))
                continue
            if m_lt:
                out['dur_max'] = int(m_lt.group(1)) - 1
                continue

        # 3. Tenta reconhecer gêneros por palavra-chave (sem prefixo)
        is_genero = False
        for gen in generos_conhecidos:
            if low == gen or f' {gen}' in f' {low}' or f'{gen} ' in f'{low} ':
                if low not in out['generos']:
                    out['generos'].append(part.strip())
                is_genero = True
                break

        if is_genero:
            continue

        # 4. Se df_filmes é fornecido, tenta reconhecer atores/diretores conhecidos
        if df_filmes is not None:
            # Busca em atores
            if 'atores' in df_filmes.columns:
                atores_unicos = set()
                for ators_str in df_filmes['atores'].dropna().unique():
                    for a in str(ators_str).split('|'):
                        atores_unicos.add(_normalize(a.strip()))
                
                norm_part = _normalize(part)
                if norm_part in atores_unicos:
                    out['atores'].append(part.strip())
                    continue

            # Busca em diretores
            if 'diretor' in df_filmes.columns:
                diretores_unicos = set()
                for d in df_filmes['diretor'].dropna().unique():
                    diretores_unicos.add(_normalize(str(d).strip()))
                
                norm_part = _normalize(part)
                if norm_part in diretores_unicos:
                    out['diretores'].append(part.strip())
                    continue

        # 5. Se nada se aplica, trata como token genérico (busca livre)
        out['text_tokens'].append(part.strip())

    return out
def _parse_filters(filter_text):
    """Parse filters - wrapper for backward compatibility. Uses the smart parser."""
    return _smart_parse_filters(filter_text)

def _parse_filters_legacy(filter_text):
    out = {
        'diretores': [],
        'atores': [],
        'generos': [],
        'text_tokens': [],
        'dur_min': None,
        'dur_max': None,
    }

    if not filter_text or not isinstance(filter_text, str):
        return out

    parts = [p.strip() for p in re.split(r'[;,\n]+', filter_text) if p.strip()]
    for part in parts:
        low = part.lower()

        # Diretor
        if low.startswith('diretor:') or low.startswith('director:'):
            name = part.split(':', 1)[1].strip()
            if name:
                out['diretores'].append(name)
            continue

        # Ator(s)
        if low.startswith('ator:') or low.startswith('atores:'):
            names = part.split(':', 1)[1].strip()
            for n in re.split(r'[,/]+', names):
                n = n.strip()
                if n:
                    out['atores'].append(n)
            continue

        # Genero
        if low.startswith('genero:') or 'gênero' in low or low.startswith('generos:') or low.startswith('genero:'):
            if ':' in part:
                vals = part.split(':', 1)[1]
                for g in re.split(r'[,/]+', vals):
                    g = g.strip()
                    if g:
                        out['generos'].append(g)
                continue

        # Duracao patterns
        if 'durac' in low or 'duraç' in low or low.startswith('min ') or re.search(r'\b\d{2,3}\b', low):
            # look for expressions like 90-120, >=90, >90, <=120, <120
            # explicit 'duracao' takes precedence
            if 'durac' in low or 'duraç' in low:
                if '-' in low:
                    m = re.search(r'(\d{1,4})\s*-\s*(\d{1,4})', low)
                    if m:
                        out['dur_min'] = int(m.group(1))
                        out['dur_max'] = int(m.group(2))
                        continue
                # relational ops
                m_ge = re.search(r'>=\s*(\d{1,4})', low)
                m_gt = re.search(r'>\s*(\d{1,4})', low)
                m_le = re.search(r'<=\s*(\d{1,4})', low)
                m_lt = re.search(r'<\s*(\d{1,4})', low)
                if m_ge:
                    out['dur_min'] = int(m_ge.group(1))
                    continue
                if m_gt:
                    out['dur_min'] = int(m_gt.group(1)) + 1
                    continue
                if m_le:
                    out['dur_max'] = int(m_le.group(1))
                    continue
                if m_lt:
                    out['dur_max'] = int(m_lt.group(1)) - 1
                    continue

            # fallback: if the part is just a number or contains minutes, treat as minimum duration
            mnum = re.findall(r'(\d{2,4})', low)
            if mnum:
                out['dur_min'] = int(mnum[0])
                continue

        # Generic token: could be actor/director/genre without prefix
        out['text_tokens'].append(part.strip())

    return out


def recomendar_filmes_hibrido(titulo_base, numero_recomendacoes=10, filtros_texto=None):
    """
    Sistema de Recomendação Híbrida:
    1. Geração de Candidatos via PNL (Similaridade de Conteúdo).
    2. Pontuação/Filtragem via Árvore de Decisão (Predição de Preferência).
    """
    global DF_FILMES, COSINE_SIM, MODELO_ARVORE
    
    _load_models()

    if DF_FILMES is None or COSINE_SIM is None:
        return "Erro: Modelos PNL não carregados. Execute o treinamento primeiro (main.py)."
    
    # 1. GERAÇÃO DE CANDIDATOS (PNL - Similaridade de Conteúdo)
    # Pega mais filmes (ex: Top 50) para ter uma base de filtragem robusta
    candidatos_pnl = pnl_module.obter_similares(
        DF_FILMES, COSINE_SIM, titulo_base, top_n=50 
    )
    
    if candidatos_pnl.empty:
        return f"Nenhum filme similar a '{titulo_base}' encontrado pelo PNL."

    # Aplicar filtros adicionais vindos do chat (diretor, atores, generos, duracao, etc.)
    parsed = _smart_parse_filters(filtros_texto, DF_FILMES)

    # Start with all True mask
    if parsed and any([parsed['diretores'], parsed['atores'], parsed['generos'], parsed['text_tokens'], parsed['dur_min'], parsed['dur_max']]):
        mask = pd.Series(True, index=candidatos_pnl.index)

        # Diretores (normalized matching)
        if parsed['diretores'] and 'diretor' in candidatos_pnl.columns:
            m = pd.Series(False, index=candidatos_pnl.index)
            norm_dir_col = candidatos_pnl['diretor'].apply(_normalize)
            for d in parsed['diretores']:
                norm_d = _normalize(d)
                m = m | norm_dir_col.str.contains(norm_d, case=False, na=False, regex=False)
            mask = mask & m

        # Atores (normalized matching)
        if parsed['atores'] and 'atores' in candidatos_pnl.columns:
            m = pd.Series(False, index=candidatos_pnl.index)
            norm_ator_col = candidatos_pnl['atores'].apply(_normalize)
            for a in parsed['atores']:
                norm_a = _normalize(a)
                m = m | norm_ator_col.str.contains(norm_a, case=False, na=False, regex=False)
            mask = mask & m

        # Generos (normalized matching)
        if parsed['generos'] and 'generos' in candidatos_pnl.columns:
            m = pd.Series(False, index=candidatos_pnl.index)
            norm_gen_col = candidatos_pnl['generos'].apply(_normalize)
            for g in parsed['generos']:
                norm_g = _normalize(g)
                m = m | norm_gen_col.str.contains(norm_g, case=False, na=False, regex=False)
            mask = mask & m

        # Texto tokens (search across title/diretor/atores/generos/sinopse when available, normalized)
        if parsed['text_tokens']:
            m = pd.Series(False, index=candidatos_pnl.index)
            norm_tit_col = candidatos_pnl['titulo'].apply(_normalize)
            norm_dir_col = candidatos_pnl['diretor'].apply(_normalize) if 'diretor' in candidatos_pnl.columns else None
            norm_ator_col = candidatos_pnl['atores'].apply(_normalize) if 'atores' in candidatos_pnl.columns else None
            norm_gen_col = candidatos_pnl['generos'].apply(_normalize) if 'generos' in candidatos_pnl.columns else None
            norm_sin_col = candidatos_pnl['sinopse'].apply(_normalize) if 'sinopse' in candidatos_pnl.columns else None
            
            for tok in parsed['text_tokens']:
                norm_tok = _normalize(tok)
                m = m | norm_tit_col.str.contains(norm_tok, case=False, na=False, regex=False)
                if norm_dir_col is not None:
                    m = m | norm_dir_col.str.contains(norm_tok, case=False, na=False, regex=False)
                if norm_ator_col is not None:
                    m = m | norm_ator_col.str.contains(norm_tok, case=False, na=False, regex=False)
                if norm_gen_col is not None:
                    m = m | norm_gen_col.str.contains(norm_tok, case=False, na=False, regex=False)
                if norm_sin_col is not None:
                    m = m | norm_sin_col.str.contains(norm_tok, case=False, na=False, regex=False)
            mask = mask & m

        # Duracao
        if parsed['dur_min'] is not None and 'duracao' in candidatos_pnl.columns:
            mask = mask & (pd.to_numeric(candidatos_pnl['duracao'], errors='coerce').fillna(0) >= parsed['dur_min'])
        if parsed['dur_max'] is not None and 'duracao' in candidatos_pnl.columns:
            mask = mask & (pd.to_numeric(candidatos_pnl['duracao'], errors='coerce').fillna(0) <= parsed['dur_max'])

        candidatos_pnl = candidatos_pnl[mask]

        if candidatos_pnl.empty:
            return f"Nenhum filme similar a '{titulo_base}' passou pelos filtros fornecidos: '{filtros_texto}'."

    # 2. PONTUAÇÃO (Árvore de Decisão - Predição de Preferência)
    
    # Calcula a probabilidade de o usuário gostar do filme (Score da Árvore)
    candidatos_pnl['score_arvore'] = decision_tree.prever_preferencia(
        MODELO_ARVORE, candidatos_pnl
    )

    # 3. COMBINAÇÃO E RECOMENDAÇÃO FINAL
    
    # Combina as pontuações: PNL * ÁRVORE. 
    # Isso garante que apenas filmes *similares* (alto PNL) E com *alta chance de ser gostado* (alto Árvore) subam no ranking.
    candidatos_pnl['score_final'] = (
        candidatos_pnl['similaridade'] * candidatos_pnl['score_arvore']
    )
    
    # Ordena e seleciona os top N
    recomendacoes_finais = candidatos_pnl.sort_values(
        by='score_final', ascending=False
    ).head(numero_recomendacoes)
    
    # Formatação do resultado (inclui coluna de atores)
    resultado = recomendacoes_finais[[
        'titulo', 'generos', 'diretor', 'atores', 'duracao', 'score_final'
    ]].reset_index(drop=True)
    
    resultado.columns = ['Título', 'Gênero(s)', 'Diretor', 'Atores', 'Duração (min)', 'Score Híbrido']
    
    return resultado.to_dict('records')


def descobrir_filmes_por_filtros(filtros_texto, numero_recomendacoes=10):
    """
    Descobrir filmes por filtros sem precisa de um título base.
    Útil para usuários que não sabem o que querem assistir.
    
    Filtra toda a base de dados por critérios (diretor, ator, gênero, duração, etc)
    e depois pontua com a Árvore de Decisão, ordenando por score final.
    """
    global DF_FILMES, COSINE_SIM, MODELO_ARVORE
    
    _load_models()

    if DF_FILMES is None:
        return "Erro: Dados de filmes não carregados. Execute o treinamento primeiro (main.py)."
    
    if not filtros_texto or not filtros_texto.strip():
        return "Por favor, forneça pelo menos um filtro (diretor, ator, gênero, duração, etc)."
    
    # Aplicar filtros na base toda
    parsed = _smart_parse_filters(filtros_texto, DF_FILMES)
    
    if not any([parsed['diretores'], parsed['atores'], parsed['generos'], parsed['text_tokens'], parsed['dur_min'], parsed['dur_max']]):
        return "Nenhum filtro válido foi fornecido. Tente: 'Tom Hanks' ou 'Acao' ou 'duracao: 90-120'."
    
    # Copiar DataFrame inteiro como candidatos
    candidatos = DF_FILMES.copy()
    mask = pd.Series(True, index=candidatos.index)
    
    # Diretores (normalized matching)
    if parsed['diretores'] and 'diretor' in candidatos.columns:
        m = pd.Series(False, index=candidatos.index)
        norm_dir_col = candidatos['diretor'].apply(_normalize)
        for d in parsed['diretores']:
            norm_d = _normalize(d)
            m = m | norm_dir_col.str.contains(norm_d, case=False, na=False, regex=False)
        mask = mask & m

    # Atores (normalized matching)
    if parsed['atores'] and 'atores' in candidatos.columns:
        m = pd.Series(False, index=candidatos.index)
        norm_ator_col = candidatos['atores'].apply(_normalize)
        for a in parsed['atores']:
            norm_a = _normalize(a)
            m = m | norm_ator_col.str.contains(norm_a, case=False, na=False, regex=False)
        mask = mask & m

    # Generos (normalized matching)
    if parsed['generos'] and 'generos' in candidatos.columns:
        m = pd.Series(False, index=candidatos.index)
        norm_gen_col = candidatos['generos'].apply(_normalize)
        for g in parsed['generos']:
            norm_g = _normalize(g)
            m = m | norm_gen_col.str.contains(norm_g, case=False, na=False, regex=False)
        mask = mask & m

    # Texto tokens (search across title/diretor/atores/generos/sinopse when available, normalized)
    if parsed['text_tokens']:
        m = pd.Series(False, index=candidatos.index)
        norm_tit_col = candidatos['titulo'].apply(_normalize)
        norm_dir_col = candidatos['diretor'].apply(_normalize) if 'diretor' in candidatos.columns else None
        norm_ator_col = candidatos['atores'].apply(_normalize) if 'atores' in candidatos.columns else None
        norm_gen_col = candidatos['generos'].apply(_normalize) if 'generos' in candidatos.columns else None
        norm_sin_col = candidatos['sinopse'].apply(_normalize) if 'sinopse' in candidatos.columns else None
        
        for tok in parsed['text_tokens']:
            norm_tok = _normalize(tok)
            m = m | norm_tit_col.str.contains(norm_tok, case=False, na=False, regex=False)
            if norm_dir_col is not None:
                m = m | norm_dir_col.str.contains(norm_tok, case=False, na=False, regex=False)
            if norm_ator_col is not None:
                m = m | norm_ator_col.str.contains(norm_tok, case=False, na=False, regex=False)
            if norm_gen_col is not None:
                m = m | norm_gen_col.str.contains(norm_tok, case=False, na=False, regex=False)
            if norm_sin_col is not None:
                m = m | norm_sin_col.str.contains(norm_tok, case=False, na=False, regex=False)
        mask = mask & m

    # Duracao
    if parsed['dur_min'] is not None and 'duracao' in candidatos.columns:
        mask = mask & (pd.to_numeric(candidatos['duracao'], errors='coerce').fillna(0) >= parsed['dur_min'])
    if parsed['dur_max'] is not None and 'duracao' in candidatos.columns:
        mask = mask & (pd.to_numeric(candidatos['duracao'], errors='coerce').fillna(0) <= parsed['dur_max'])

    candidatos = candidatos[mask]
    
    if candidatos.empty:
        return f"Nenhum filme encontrado com os filtros fornecidos: '{filtros_texto}'."
    
    # Pontuação via Árvore de Decisão
    candidatos['score_arvore'] = decision_tree.prever_preferencia(
        MODELO_ARVORE, candidatos
    )
    
    # Ordenar por score da árvore (sem PNL, pois não há similaridade aqui)
    recomendacoes_finais = candidatos.sort_values(
        by='score_arvore', ascending=False
    ).head(numero_recomendacoes)
    
    # Formatação do resultado
    resultado = recomendacoes_finais[[
        'titulo', 'generos', 'diretor', 'atores', 'duracao', 'score_arvore'
    ]].reset_index(drop=True)
    
    resultado.columns = ['Título', 'Gênero(s)', 'Diretor', 'Atores', 'Duração (min)', 'Score de Preferência']
    
    return resultado.to_dict('records')

# Nota: As funções busca_filme.py e coleta_api.py devem ser adaptadas para 
# garantir que os dados de entrada (principalmente a 'duração') estejam corretos.