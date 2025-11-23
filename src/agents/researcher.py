from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import AgentState
from src.tools.search import get_google_search_tool
from src.config import config as app_config
from src.utils.logger import logger

# Initialize models
# We will initialize inside the node for dynamic selection
# gemini_model = ChatVertexAI(model_name=config.RESEARCHER_MODEL, temperature=0)

# Tools
search_tool = get_google_search_tool()
tools = [search_tool]
# model_with_tools = gemini_model.bind_tools(tools)

# Researcher Prompt
RESEARCHER_SYSTEM_PROMPT = """You are an expert researcher. Your goal is to find comprehensive information about the topic.
You have access to a Google Search tool.
Your research must cover:
1. **Biographical & Professional**: Career history, education, known associates.
2. **Financial**: Business holdings, investments, bankruptcies, fraud allegations.
3. **Legal & Regulatory**: Lawsuits, regulatory actions, criminal records.
4. **Behavioral**: Public controversies, statements, reputation.

Analyze the current findings and generate new search queries to fill in gaps in these specific areas.
You should perform multiple searches to get a deep understanding.
29. **Source Verification**: ALWAYS try to verify important claims across multiple sources. If a fact is disputed, note the conflict.
If you have enough information or have reached the iteration limit, summarize the findings and stop searching.
"""

from langchain_core.runnables import RunnableConfig

def researcher_node(state: AgentState, config: RunnableConfig):
    """
    The researcher agent node.
    """
    try:
        topic = state["research_topic"]
        findings = state["findings"]
        iterations = state["iterations"]
        search_queries = state["search_queries"]
        
        # Dynamic Model Selection
        model_name = state.get("researcher_model", app_config.RESEARCHER_MODEL)
        logger.info(f"Researcher Node: Using model '{model_name}'")
        
        if model_name.startswith("gpt"):
            model = ChatOpenAI(model=model_name, temperature=0)
        else:
            model = ChatVertexAI(model_name=model_name, temperature=0)
            
        # Check iteration limit
        if iterations > 3:
            logger.info("Researcher Node: Iteration limit reached. Forcing summary.")
            # Do NOT bind tools. Force the model to summarize.
            # Update prompt to be explicit about stopping.
            messages = [
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT + "\n\nCRITICAL: You have reached the iteration limit. You MUST NOT search anymore. Summarize the findings immediately."),
                HumanMessage(content=f"Topic: {topic}\nCurrent Findings: {findings}\nPrevious Queries: {search_queries}\nIterations: {iterations}")
            ]
            response = model.invoke(messages, config=config)
        else:
            model_with_tools = model.bind_tools(tools)
            logger.info(f"Researcher Node: Iteration {iterations} for topic '{topic}'")
            
            messages = [
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(content=f"Topic: {topic}\nCurrent Findings: {findings}\nPrevious Queries: {search_queries}\nIterations: {iterations}")
            ]
            
            response = model_with_tools.invoke(messages, config=config)
        logger.debug(f"Researcher Response: {response.content[:100]}...")
            
        return {"messages": [response], "iterations": iterations + 1}
        
    except Exception as e:
        logger.error(f"Error in researcher_node: {e}", exc_info=True)
        raise
