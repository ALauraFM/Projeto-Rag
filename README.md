# YAITEC RAG Assistant

Assistente RAG sobre a base de conhecimento do **YAITEC Atende**, construído com Python + FastAPI + LangChain + Qdrant + OpenAI.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.11, FastAPI |
| RAG | LangChain (LCEL) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Voz (STT/TTS) | OpenAI `whisper-1` + `gpt-4o-mini-tts` |
| Vector Store | Qdrant (Docker) |
| Frontend | HTML/JS estático servido pelo FastAPI |
| Containers | Docker Compose |

---

## Decisões de arquitetura

- **LCEL chain** — uso da LangChain Expression Language para composição do pipeline RAG, evitando as APIs depreciadas (`RetrievalQA`).
- **Qdrant** — vector store com persistência em volume Docker; a coleção é criada automaticamente na primeira execução.
- **Ingestão idempotente** — `app/ingest.py` verifica se a coleção já tem pontos antes de re-embedar; roda uma vez no start do container.
- **Citação de fontes** — o nome do arquivo de origem é injetado no prompt e devolvido no JSON da resposta (`"fontes"`).
- **Chunk size 500 / overlap 50** — equilibra contexto suficiente por chunk sem exceder o context window do modelo.

---

## Ferramentas de IA utilizadas

- **Windsurf (Cascade)** — revisão de código
- **OpenAI GPT-4o-mini** — LLM de resposta em produção
- **OpenAI text-embedding-3-small** — embeddings para o índice RAG

---

## Pré-requisitos

- Docker e Docker Compose instalados
- Chave de API da OpenAI

---

## Como rodar

### 1. Clone e configure

```bash
git clone <url-do-repositorio>
cd yaitec_desafio
cp .env.example .env
# Edite .env e coloque sua OPENAI_API_KEY
```

### 2. Suba com Docker Compose

```bash
docker compose up --build
```

Na primeira execução, o container `api` vai:
1. Popular o Qdrant com os chunks dos documentos em `/docs`
2. Iniciar o servidor FastAPI na porta `8000`

### 3. Acesse a interface

Abra [http://localhost:8000](http://localhost:8000) no navegador.

---

## Exemplo com curl

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Quais são os planos disponíveis e seus preços?"}'
```

Resposta esperada:

```json
{
  "resposta": "Os planos do YAITEC Atende são:\n- **Starter**: R$ 299/mês, 1 agente, até 1.000 conversas. [Fonte: planos-e-precos.md]\n- **Pro**: R$ 899/mês, 5 agentes, até 10.000 conversas. [Fonte: planos-e-precos.md]\n- **Enterprise**: sob consulta, agentes ilimitados. [Fonte: planos-e-precos.md]",
  "fontes": ["planos-e-precos.md"]
}
```

---

## Rodar testes

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Interface web (chat) |
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Pergunta ao assistente RAG (texto) |
| `POST` | `/ask-audio` | Pergunta por áudio (STT → RAG → TTS) |
| `GET` | `/docs` | Swagger UI |

---

## Diferencial: Voz (STT/TTS)

O assistente aceita perguntas por **áudio** e responde também em **áudio**:

- **STT**: OpenAI Whisper (`whisper-1`) transcreve o áudio enviado
- **TTS**: OpenAI (`gpt-4o-mini-tts`, voz `alloy`) sintetiza a resposta

Na interface, clique no botão de **microfone**, fale sua pergunta e clique novamente para enviar. A resposta é exibida em texto (com fontes) e reproduzida em áudio automaticamente.

O endpoint `POST /ask-audio` recebe um arquivo de áudio (`multipart/form-data`, campo `file`) e retorna:

```json
{
  "pergunta": "transcrição da pergunta",
  "resposta": "resposta fundamentada [Fonte: ...]",
  "fontes": ["planos-e-precos.md"],
  "audio_base64": "<mp3 em base64>"
}
```

---

## Estrutura do projeto

```
yaitec_desafio/
├── docs/                        # Base de conhecimento (3 arquivos .md)
├── app/
│   ├── config.py                # Configurações via pydantic-settings
│   ├── retrieval_pipeline.py    # Embeddings, Qdrant e recuperação de contexto
│   ├── rag.py                   # Geração: prompt + LLM + composição da chain (LCEL)
│   ├── ingestion_pipeline.py    # Carregamento, chunking e ingestão no Qdrant
│   ├── voice.py                 # STT (Whisper) e TTS via OpenAI
│   └── main.py                  # FastAPI: /ask, /ask-audio, health, static
├── static/
│   └── index.html               # Chat UI
├── tests/
│   └── test_ask.py              # Testes do endpoint /ask
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
