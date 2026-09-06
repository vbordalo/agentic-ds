# Agentic AI for Data Science

## Instalando Ollama

```shell
Conda environment
    Python
      │
      │ HTTP/API
      ▼
   Ollama
      │
      ▼
Qwen2.5 1.5B
      │
      ▼
NVIDIA P1000
```

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

## Baixando e rodando `Qwen2.5 1.5B` (~986 MB)

```shell
ollama run qwen2.5:1.5b
```

## Benchmarking

```shell
request
  │
  ├── model load
  │
  ├── prompt evaluation
  │
  ├── token generation
  │
  ▼
response
```

```shell
python src/agentic_ds/llm_benchmark.py

--- RESPONSE ---

Exploratory Data Analysis (EDA) is a critical first step in data science where one examines and understands the structure and patterns within a dataset. It involves using graphical methods and statistical summaries to explore the data, identifying trends, anomalies, and relationships that might be overlooked in the initial stages. EDA is useful because it helps in making informed decisions by providing insights that are crucial for building a robust data-driven model.
```
Cold start:
```shell
--- METRICS ---

Model:                qwen2.5:1.5b
Total duration:       5.286 s
Model load duration:  2.363 s
Prompt eval duration: 0.109 s
Generation duration:  2.811 s
Prompt tokens:        41
Generated tokens:     84
Generation speed:     29.89 tokens/s
```

Warm inference:
```shell
--- METRICS ---

Model:                qwen2.5:1.5b
Total duration:       2.987 s
Model load duration:  0.001 s
Prompt eval duration: 0.160 s
Generation duration:  2.816 s
Prompt tokens:        41
Generated tokens:     84
Generation speed:     29.83 tokens/s
```

