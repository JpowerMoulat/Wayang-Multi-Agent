# AI-Wayang-Multi
A multi-agent system for translating natural-language queries into executable Apache Wayang plans.

The system is implemented as a LangGraph pipeline orchestrated via a CLI interface. Six specialized agents (Specifier, Selector, Decomposer, Builder, Refiner, Debugger) collaborate to produce and execute a Wayang JSON plan against a running Apache Wayang JSON-REST API server.

The system was developed as part of the Research Project **"Comparing Agentic AI Frameworks for Natural Language
Wayang Queries"**. For a deeper understanding of the system, we recommend reading the report.

# Start
Run the pipeline interactively:

```bash
python main.py
```

You will be prompted to enter a natural-language query. The system will process it through the agent pipeline and report the result.

# Requirements
The following components are required to run the system:

- A running Apache Wayang JSON-REST API server
- An OpenAI API Key
- Python with listed dependencies (see `requirements.txt` or `pyproject.toml`)

# Setup

## Python and Python Dependencies
Python 3.12.4 is recommended, as this version has been tested.

Dependencies can be installed via UV (recommended):

```bash
uv sync
```

Or via pip:

```bash
pip install -r requirements.txt
```

## .env File
You must create a `.env` file in the project root with the following variables.

**Required**

```
OPENAI_API_KEY=        # API key for OpenAI LLMs
WAYANG_URL=            # URL to the Wayang JSON-REST server

JDBC_URI=              # JDBC connection string
JDBC_USERNAME=         # Database username
JDBC_PASSWORD=         # Database password
```

OR/AND (at least one data source must be configured):

```
INPUT_FOLDER=          # Path to folder containing .txt input files
```

**Optional**

```
LOG_FOLDER=            # Path for session log output (JSON files)
OUTPUT_FOLDER=         # Preferred output location for result files

SPECIFIER_LLM=         # GPT model for Specifier Agent
SPECIFIER_REASON_EFFORT=

SELECTOR_LLM=          # GPT model for Selector Agent
SELECTOR_REASON_EFFORT=

DECOMPOSER_LLM=        # GPT model for Decomposer Agent
DECOMPOSER_REASON_EFFORT=

BUILDER_LLM=           # GPT model for Builder Agent
BUILDER_REASON_EFFORT=

REFINER_LLM=           # GPT model for Refiner Agent
REFINER_REASON_EFFORT=

USE_DEBUGGER=          # True/False — enable automatic plan debugging
DEBUGGER_LLM=          # GPT model for Debugger Agent
DEBUGGER_REASON_EFFORT=
MAX_ITR=               # Max debug iterations (default: 5)
```

# Recommendation
We recommend pre-generating schemas for your data sources before running queries. This can be done by running the schema loader utility directly:

```bash
python -c "from ai_wayang_multi.utils.schema_loader import SchemaLoader; SchemaLoader().load_all()"
```
