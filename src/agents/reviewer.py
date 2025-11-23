from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.config import config as app_config
from src.utils.logger import logger
import json
from langchain_core.runnables import RunnableConfig

REVIEWER_SYSTEM_PROMPT = """You are an expert Quality Control Reviewer for an intelligence agency.
Your goal is to strictly evaluate the Analyst's report against the gathered findings.

You must check for:
1. **Hallucinations**: Are the claims in the report supported by the provided findings?
2. **Logic & Consistency**: Is the Risk Score justified by the rubric? Are there contradictions?
3. **Completeness**: Did the agent answer the user's research topic?
4. **Source Verification**: Are key facts backed by sources?

Output your decision in the following format:
DECISION: [APPROVE | REJECT]
FEEDBACK: [Detailed feedback explaining why, or "Looks good" if approved.]

If you REJECT, be specific about what needs to be fixed (e.g., "Claim X is not in findings", "Risk score 8 is too high for minor issues").
"""

def reviewer_node(state: AgentState, config: RunnableConfig):
    """
    The reviewer agent node.
    """
    try:
        topic = state["research_topic"]
        findings = state["findings"]
        report = state["report"]
        risk_score = state["risk_score"]
        
        model_name = state.get("analyst_model", app_config.ANALYST_MODEL) 
        logger.info(f"Reviewer Node: Using model '{model_name}'")
        
        if model_name.startswith("gpt"):
            model = ChatOpenAI(model=model_name, temperature=0)
        else:
            model = ChatVertexAI(model_name=model_name, temperature=0)
        
        findings_text = "\n".join(findings)
        
        if model_name.startswith("gpt"):
            if len(findings_text) > 20000:
                logger.warning("Reviewer Node: Used GPT, truncating to 20k chars.")
                findings_text = findings_text[:20000] + "... [TRUNCATED]"
        else:
            logger.info("Reviewer Node : Used Gemini no truncation ")
        
        key_facts = state.get("key_facts", [])
        key_facts_text = json.dumps(key_facts, indent=2) if key_facts else "None provided"

        messages = [
                SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                HumanMessage(content=f"""Topic: {topic}
                Risk Score: {risk_score}
                Report:
                {report}
                
                Key Facts:
                {key_facts_text}

                Findings (Truncated if too long):
                {findings_text}
                """)
            ]
        
        response = model.invoke(messages, config=config)
        content = response.content
        
        logger.info(f"Reviewer Decision: {content}")
        
        return {"reviewer_feedback": content}
        
    except Exception as e:
        logger.error(f"Error in reviewer_node: {e}", exc_info=True)
        raise
