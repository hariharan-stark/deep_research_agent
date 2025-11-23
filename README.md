# 🚀 Deep Research Agent - Setup Guide

This guide provides step-by-step instructions to set up and run the Deep Research Agent.

## 📋 Prerequisites

- **Python 3.10+** installed on your system.
- **Google Cloud Project** with Vertex AI API enabled.
- **OpenAI API Key** (for GPT-4/4o support).
- **SerpApi Key** (for Google Search).

## 🛠️ Installation

1.  **Clone the Repository** (if applicable) or navigate to the project directory.

2.  **Create a Virtual Environment** (Recommended):

    ```bash
    python -m venv ai_env
    # Windows
    ai_env\Scripts\activate
    # Mac/Linux
    source ai_env/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: Ensure `chainlit`, `langgraph`, `langchain-google-vertexai`, `langchain-openai`, and `google-search-results` are in your `requirements.txt`._

## 🔑 Configuration (.env)

Create a `.env` file in the root directory of the project. You can copy the structure below:

```env
# Google Cloud / Vertex AI
GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
GOOGLE_CLOUD_PROJECT="your-project-id"
LOCATION="us-central1" # or your preferred region

# Search Tool
SERPAPI_API_KEY="your-serpapi-key"

# OpenAI (Optional, for GPT models)
OPENAI_API_KEY="sk-..."

# LangSmith (Optional, for tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="your-langchain-api-key"
```

### ⚠️ Important Notes:

- **Vertex AI**: Ensure your service account has the "Vertex AI User" role.
- **SerpApi**: Required for the agent to perform live Google searches.

## 🏃‍♂️ Running the Application

The application uses **Chainlit** for the UI. To start the server:

```bash
chainlit run app.py -w
```

- `-w`: Enables auto-reload (watch mode) for development.

Once running, the UI will be accessible at `http://localhost:8000`.

## 🖥️ Usage

1.  **Select Models**: Use the settings panel to choose your **Researcher** (e.g., `gemini-2.5-pro`) and **Analyst** (e.g., `gpt-4o`) models.
2.  **Start Research**: Type the name of a person or company in the chat (e.g., "Elon Musk", "Apple Inc.").
3.  **Monitor Progress**: Watch the agent perform research, analyze findings, and undergo quality review.
4.  **View Report**: The final risk assessment and intelligence report will be displayed once the process is complete.
