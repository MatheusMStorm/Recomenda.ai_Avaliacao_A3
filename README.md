# 🎬 Recomenda.ai - Sistema de Recomendação Híbrida

> **Avaliação A3 - Inteligência Artificial**

O **Recomenda.ai** é um sistema inteligente de recomendação de filmes que utiliza uma abordagem híbrida, combinando **Processamento de Linguagem Natural (PNL)** para análise de similaridade de conteúdo e **Árvores de Decisão** para filtragem e previsão de preferência do utilizador.

<div align="center">
  <img src="img/Recomenda.ai.png" alt="Logo Recomenda.ai" width=200 height=200>
</div>

## 📋 Sobre o Projeto

Este projeto foi desenvolvido como parte da Avaliação A3. O sistema permite descobrir filmes de duas maneiras:
1.  **Por Filme Similar:** Busca obras com sinopse e características similares via PNL (TF-IDF e Cosine Similarity).
2.  **Por Filtros Inteligentes:** Interpreta interesses em linguagem natural (ex: "Tom Hanks, Ação, curta duração") e utiliza uma Árvore de Decisão para classificar as recomendações.

## 🚀 Guia de Instalação e Execução (Passo a Passo)

O projeto pode ser executado de duas formas: na nuvem (sem instalação) ou localmente.

### ☁️ Opção 1: GitHub Codespaces (Recomendado para o Avaliador)

Esta é a forma mais rápida de testar, pois o ambiente já vem configurado na nuvem.

1.  No topo deste repositório, clique no botão verde **<> Code**.
2.  Selecione a aba **Codespaces**.
3.  Clique no botão verde **Create codespace on main**.

**No terminal do Codespaces (parte inferior da tela), execute:**

```bash
# 1. Instalar as dependências
pip install -r requirements.txt

# 2. TREINAR OS MODELOS (⚠️ Passo Obrigatório)
# Este comando gera os arquivos .pkl necessários para a IA funcionar.
python main.py

# 3. Rodar a aplicação
python -m streamlit run app.py
O sistema irá notificar que a aplicação está rodando na porta 8501. Clique em "Open in Browser".
```

### 💻 Opção 2: Rodar Localmente

Caso prefira rodar na sua máquina, siga os passos abaixo no terminal (Git Bash, PowerShell ou Terminal):

1. Clonar o Repositório
```bash
git clone [https://github.com/matheusmstorm/recomenda.ai_avaliacao_a3.git](https://github.com/matheusmstorm/recomenda.ai_avaliacao_a3.git)
cd recomenda.ai_avaliacao_a3
```

2. Configurar o Ambiente (Recomendado)
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
.\\venv\\Scripts\\activate

# Ativar (Mac/Linux)
source venv/bin/activate
```

3. Instalar Dependências
```bash
pip install -r requirements.txt
```

## 4. Treinar os Modelos de IA (⚠️ Importante)
Antes de abrir o site, é necessário processar os dados e criar a árvore de decisão.
```bash
python main.py
Aguarde a mensagem: --- Treinamento Concluído com Sucesso ---
```

## 5. Executar a Interface
```bash
python -m streamlit run app.py
```
## 🖼️ Localização dos Banners e Documentação dos Grupos
--------------------------------------------------
Para facilitar a avaliação, todos os materiais visuais e documentos produzidos pelo Grupo – Módulo PNL estão organizados na pasta:

👉 Clique aqui para acessar os banners e documentos — [texto](./docs/grupo_pnl/)


## 📝 Nota para o Avaliador (Resolução de Problemas)
--------------------------------------------------

Reunimos aqui os principais pontos de atenção para a correção:

## Erros Comuns e Soluções

### 🔍 **Erro "Modelos de IA não encontrados"**
----------------------------------------
Se ao abrir o streamlit run app.py aparecer uma mensagem vermelha informando falta de modelos:
1. Pare a execução atual (Ctrl+C)
2. Execute `python main.py` para gerar os arquivos na pasta model/
3. Tente novamente executar a aplicação

## 🛠️ Tecnologias Utilizadas
------------------------
- Python 3.12
- Streamlit: Front-end interativo
- Scikit-learn: Árvore de Decisão e Vetorização de Texto
- Pandas & NumPy: Processamento de dados

## ✒️ **Equipe de Desenvolvimento**
------------------------------
- João Fernandes
- Maria Eduarda
- Marlon Deivide
- Matheus Moura
- Michel Silva
- Milena Silva
- Pablo Anderson
