# Agentic Data Science

A small, framework-free agentic data science project built to explore how an LLM can plan, call deterministic tools, maintain explicit state, validate actions, and synthesize a final answer.

The current case study uses the **UCI Communities and Crime** dataset and focuses on the early stages of a data science workflow: dataset inspection, data quality checks, and descriptive analysis.

The project is intentionally implemented without agent frameworks such as LangChain or LangGraph. The goal is to make the mechanics of planning, tool calling, orchestration, state management, validation, and synthesis explicit.

## Current Architecture

```text
User
  |
  v
Planner
  |
  v
Agent Runtime / Executor
  |
  +--> Tool call
  |      |
  |      v
  |   Deterministic Python tool
  |      |
  |      v
  |   Update AgentState
  |      |
  |      v
  |   remaining_tasks?
  |      |
  |      +--> yes --> Executor
  |      |
  |      +--> no  --> Final Synthesizer
  |
  v
Final answer
```

The main modules are:

```text
src/agentic_ds/
├── agent_runtime.py
├── agent_state.py
├── planner.py
├── synthesizer.py
└── data_tools.py
```

### `agent_runtime.py`

Main entry point and orchestrator.

Responsibilities:

- interactive CLI;
- runtime configuration through Click;
- executor loop;
- tool registry and dispatch;
- validation of tool names, arguments, and planned tasks;
- agent state transitions;
- execution logging;
- termination when all planned tasks are complete;
- invocation of the final synthesizer.

### `agent_state.py`

Defines the explicit runtime state.

The current `AgentState` tracks:

- original user request;
- planned tasks;
- completed tasks;
- remaining tasks;
- completed tools;
- tool execution results;
- current runtime step.

### `planner.py`

Contains the planning component.

The planner decomposes a user request into a minimal set of independently verifiable information needs. It does not execute tools or answer the user's question.

The planner returns structured JSON that is validated before execution begins.

### `synthesizer.py`

Contains the final response component.

Once all planned tasks have been completed, the synthesizer receives the original request and the collected agent state and generates the final natural-language answer using only the available evidence.

### `data_tools.py`

Contains deterministic data science capabilities.

Current tools:

- `inspect_dataframe(df)`
  - number of rows and columns;
  - column names;
  - dtype counts;
  - approximate memory usage.

- `inspect_missing_values(df)`
  - columns containing missing values;
  - missing count per affected column;
  - total number of missing values.

- `describe_numeric_columns(df, columns=None)`
  - count;
  - mean;
  - standard deviation;
  - minimum;
  - quartiles;
  - median;
  - maximum.

The LLM does not execute these functions directly. It produces a structured tool request, and the Python runtime validates and executes it.

## Project Structure

```text
agentic-ds/
├── data/
│   └── raw/
│       └── communities.csv
├── logs/
│   └── agent_run_YYYYMMDD_HHMMSS.log
├── artifacts/
├── src/
│   └── agentic_ds/
│       ├── agent_runtime.py
│       ├── agent_state.py
│       ├── data_tools.py
│       ├── planner.py
│       └── synthesizer.py
└── README.md
```

`artifacts/` is reserved for future generated outputs and is not currently used.

## Dataset

The current dataset is stored at:

```text
data/raw/communities.csv
```

Missing values represented by `?` in the original data are converted to `NaN` during ingestion.

The current dataset contains:

- 1,994 rows;
- 128 columns;
- 127 numeric columns after ingestion;
- target column: `ViolentCrimesPerPop`.

## Local LLM Runtime

The project currently uses [Ollama](https://ollama.com/) as the local LLM runtime.

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The current default model is:

```text
qwen2.5:3b-instruct-q4_K_M
```

Example:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

The runtime has also been tested with smaller and larger local models to study GPU memory limits, CPU offloading, latency, and model quality.

## Running the Agent

The runtime is interactive: Click configures the runtime, while the user request is entered after startup.

Example:

```bash
python src/agentic_ds/agent_runtime.py \
  --model qwen2.5:3b-instruct-q4_K_M \
  --temperature 0.0 \
  --max-steps 5
```

Startup:

```text
Examples of questions you can ask:
- How many rows and columns are in the dataset?
- Which columns contain missing values?
- Tell me the dataset dimensions and which columns contain missing values.
- Give me descriptive statistics for population and medIncome.
- What are the typical values and ranges of population and householdsize?
- Tell me the dataset dimensions, identify missing data, and summarize population and medIncome.
- Compare the descriptive statistics of population and medIncome.
- Give me a concise assessment of the dataset structure, data completeness, and the typical values of population and medIncome.

User:
```

Runtime options:

```text
--model         Ollama model used by the agent
--temperature   Sampling temperature
--max-steps     Maximum number of executor steps
```

For the current experiments, `temperature=0.0` is used to reduce sampling variability.

## Runtime Validation

The runtime does not assume that LLM output is correct.

Before executing a tool call, it validates:

```text
tool exists?
    |
arguments are valid?
    |
tasks is a non-empty list?
    |
tasks exactly match remaining_tasks?
    |
execute deterministic tool
```

Structured output is requested from Ollama using JSON mode, while Python performs additional semantic validation.

This distinction is important:

> Valid JSON is not necessarily a valid agent action.

## Execution Logs

Each runtime execution creates a timestamped log in:

```text
logs/
```

Example:

```text
logs/agent_run_20260915_231939.log
```

A log currently records:

- runtime configuration;
- model name;
- temperature;
- maximum number of steps;
- user request;
- raw planner output;
- each LLM executor decision;
- tool results;
- agent state after each successful step;
- final synthesized answer.

Example:

```text
--- RUN CONFIG ---
{
  "model": "qwen2.5:3b-instruct-q4_K_M",
  "max_steps": 5,
  "temperature": 0.0
}

--- USER REQUEST ---
How many rows and columns are in the dataset?

--- PLANNER RAW OUTPUT ---
...

--- LLM DECISION 1 ---
...

--- TOOL RESULT 1: inspect_dataframe ---
...

--- STATE AFTER STEP 1 ---
...

--- FINAL ANSWER ---
The dataset has 1994 rows and 128 columns.
```

The logs are useful for auditing agent behavior and comparing models, prompts, runtime configurations, and future architectural changes.

## LLM Benchmarking

A separate benchmark script can be used to measure local inference performance:

```bash
python src/agentic_ds/llm_benchmark.py
```

Metrics include:

- total duration;
- model load duration;
- prompt evaluation duration;
- generation duration;
- prompt tokens;
- generated tokens;
- generation speed.

Example measurements for `qwen2.5:1.5b` on the development machine:

```text
Cold start
Total duration:       5.286 s
Model load duration:  2.363 s
Generation duration:  2.811 s
Generation speed:     29.89 tokens/s

Warm inference
Total duration:       2.987 s
Model load duration:  0.001 s
Generation duration:  2.816 s
Generation speed:     29.83 tokens/s
```

These measurements are retained as a hardware/runtime baseline; the current agent defaults to the larger Qwen2.5 3B model.

## Design Principles

The project currently follows a few simple rules:

- use LLMs for semantic decisions, planning, routing, and synthesis;
- use deterministic Python tools for exact computation;
- keep agent state explicit and inspectable;
- validate LLM actions before execution;
- prefer runtime enforcement over prompt-only assumptions;
- keep planning, execution, and final synthesis as separate responsibilities;
- log the complete execution trace for debugging and evaluation.

## Current Scope

The current implementation focuses on early data science tasks:

```text
data loading
    ->
schema inspection
    ->
data quality
    ->
descriptive statistics
```

Model training is intentionally deferred.

The next major development step is to expand the deterministic tool repertoire into broader data science capabilities while keeping the runtime architecture reusable.

Possible future capabilities include:

- categorical-variable inspection;
- duplicate and constant-column detection;
- correlation analysis;
- distribution analysis;
- outlier detection;
- target-focused exploratory analysis;
- feature selection and engineering;
- model training and evaluation;
- artifact generation.

## Status

Implemented:

- local Ollama integration;
- structured planner;
- explicit `AgentState`;
- iterative tool-calling runtime;
- deterministic tool execution;
- tool argument validation;
- planned-task validation;
- execution logging;
- separate final synthesizer;
- Click runtime configuration.

In progress:

- expansion of the data science tool repertoire;
- richer final synthesis;
- runtime evaluation and benchmarking.

Planned:

- artifact generation;
- later data science workflow stages;
- broader skill composition.
