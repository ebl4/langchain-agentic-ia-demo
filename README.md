## 🚀 Setup

### Prerequisites

- [git](https://git-scm.com/install/) is recommended
- A package/project manager: [uv](https://docs.astral.sh/uv/) (recommended) or [pip](https://pypi.org/project/pip/)
- Requires Python >=3.12, <3.14  If you use `uv`, it will take care of this for you. [More info](#python-virtual-environments)

### Installation

Edit the .env file to include the keys below for [Models](#model-providers) 

- Get a Google API Key [here](https://ai.google.dev/gemini-api/docs/quickstart).
- Optional, Create a [LangSmith](https://smith.langchain.com/) account and API Key.  

```bash
# Manual installs for checking: uv

# Required
TAVILY_API_KEY='your_tavily_api_key_here'
GOOGLE_API_KEY='your_google_api_key_here'
```

Make a virtual environment and install dependencies. [More info](#python-virtual-environments)

<details open>
<summary>Using uv (recommended)</summary>

```bash
uv sync
```

</details>

<details>
<summary>Using pip</summary>

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

</details>

<details>
<summary>Environment Variable Conflicts</summary>

If you see a warning about "ENVIRONMENT VARIABLE CONFLICTS DETECTED", you have API keys set in your system environment that differ from your .env file. Since `load_dotenv()` doesn't override existing variables by default, your system values will be used.

**Solutions:**
1. Do nothing and accept the system environment variable value
2. Unset the conflicting system environment variables for this shell session (commands provided in warning)
3. Use `load_dotenv(override=True)` in your notebooks to force .env values to take precedence
4. Update your .env file or shell init so the values are in agreement

</details>

<details>

### Python Virtual Environments

Managing your Python version is often best done with virtual environments. This allows you to select a Python version for the course independent of the system Python version.

<details open>
<summary>Using uv (recommended)</summary>

`uv` will install a version of Python compatible with the versions specified in the `pyproject.toml` in the `.venv` directory when running the `uv sync` specified above. It will use this version when invoking with `uv run`. For additional information, please see [uv](https://docs.astral.sh/uv/).
</details>

<details>
<summary>Using pyenv + pip</summary>

If you are using pip instead of uv, you may prefer using pyenv to manage your Python versions. For additional information, please see [pyenv](https://github.com/pyenv/pyenv).

```bash
pyenv install 3.12
pyenv local 3.12
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

### Model Providers

If you don't have an Google API key, you can sign up [here](https://ai.google.dev/gemini-api/docs/quickstart).

Tavily is a search provider that returns search results in an LLM-friendly way. They have a generous free tier. [Tavily](https://tavily.com)
