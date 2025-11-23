from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.config import config as app_config
from src.utils.logger import logger
import re
import json


ANALYST_SYSTEM_PROMPT = """You are an expert intelligence analyst.
Your goal is to review the findings from the researcher and produce a comprehensive intelligence report.

You MUST structure your report exactly with the following sections:

1. **Deep Fact Extraction**:
   - Biographical details (verified).
   - Professional history & Financial connections.
   - Behavioral patterns.
   
2. **Risk Pattern Recognition**:
   - Flag potential red flags (fraud, legal issues, controversies).
   - Highlight inconsistencies in public records or statements.
   
3. **Connection Mapping**:
   - Trace relationships between the subject and other entities, organizations, or events.
   - Identify any hidden or non-obvious connections.
   
4. **Source Validation**:
   - List key sources used.
   - Assign a confidence score (0-10) to the overall findings.
   - Cross-reference facts and note any unverified claims.

5. **Risk Score Calculation**:
   You MUST use the following rubric to determine the Risk Score (0-10):
   - **0-2 (Low Risk)**: No significant negative findings. Standard professional profile.
   - **3-4 (Low-Medium)**: Minor controversies or unverified rumors. No legal/financial red flags.
   - **5-6 (Medium)**: Verified controversies, minor legal issues, or concerning associations.
   - **7-8 (High)**: Significant legal actions, fraud allegations, major regulatory fines, or patterns of unethical behavior.
   - **9-10 (Critical)**: Convicted crimes, sanctions, direct links to organized crime/terrorism, or massive fraud.

   **Justification**: You must explicitly state which criteria were met to justify the score.

6. **Inline Citations**:
   - You MUST cite your sources for every major claim.
   - Use the format: `Statement [Short Source Name](URL)`.
   - **CRITICAL**: The text in the brackets `[]` MUST be short (e.g., "Reuters", "nytimes.com", "Wikipedia"). DO NOT put the full URL in the brackets.
   - Example: `Elon Musk founded SpaceX in 2002 [SpaceX](https://spacex.com/history)`.
   - Use the URLs provided in the findings.

Output the final **Risk Score** clearly at the end.

7. **KEY_FACTS**:
   - At the very end of your response, output a list of the top 5-10 most critical verified facts used in your analysis.
   - Format: `KEY_FACTS: ["Fact 1", "Fact 2", ...]` (JSON style list)
"""

from langchain_core.runnables import RunnableConfig

def analyst_node(state: AgentState, config: RunnableConfig):
    """
    The analyst agent node.
    """
    try:
        findings = state["findings"]
        topic = state["research_topic"]
        revisions = state.get("revisions", 0)
        
        # Increment revisions if we are looping back 
        # but we can adjust the limit check to >= 3 or handle it in the edge)
        # Better: The edge checks state['revisions']. We should increment it here if feedback exists.
        if state.get("reviewer_feedback"):
             revisions += 1
        
        model_name = state.get("analyst_model", app_config.ANALYST_MODEL)
        logger.info(f"Analyst Node: Using model '{model_name}'")
        
        if model_name.startswith("gpt"):
            model = ChatOpenAI(model=model_name, temperature=0)
        else:
            model = ChatVertexAI(model_name=model_name, temperature=0)
        
        logger.info(f"Analyst Node: Analyzing {len(findings)} findings for '{topic}'")
        
        # Combine findings into a single string
        findings_text = "\n".join(findings)
        
        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=f"Topic: {topic}\nFindings:\n{findings_text}")
        ]
        
        response = model.invoke(messages, config=config)

        # Parse response (assuming model returns text)
        report = response.content
        logger.info("Analyst Node: Report generated successfully")
        
        risk_score = 0.0 # Default fallback
        
        # Log the end of the report to see what we are parsing
        report_tail = report[-500:] if len(report) > 500 else report
        msg = f"Analyst Node: Parsing report for Risk Score. Tail:\n{report_tail}"
        logger.info(msg)
        
        # Try multiple patterns
        patterns = [
            r"Risk Score.*?:?\s*(\d+(\.\d+)?)",
            r"Score.*?:?\s*(\d+(\.\d+)?)",
            r"Risk.*?:?\s*(\d+(\.\d+)?)/10"
        ]
        
        match = None
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.DOTALL)
            if match:
                logger.info(f"Analyst Node: Matched pattern '{pattern}'")
                break
        
        if match:
            try:
                risk_score = float(match.group(1))
                logger.info(f"Analyst Node: Extracted Risk Score: {risk_score}")
            except ValueError:
                logger.error(f"Analyst Node: Could not parse risk score from match '{match.group(1)}'")
        else:
            logger.error("Analyst Node: No Risk Score found in report using any pattern.")
            
        # Extract Key Facts
        key_facts = []
        try:
            facts_match = re.search(r"KEY_FACTS:\s*(\[.*?\])", report, re.DOTALL)
            if facts_match:
                key_facts = json.loads(facts_match.group(1))
                logger.info(f"Analyst Node: Extracted {len(key_facts)} key facts.")
                report = re.sub(r"\n*\*\*?KEY_FACTS\*\*?:.*", "", report, flags=re.DOTALL).strip()
            else:
                logger.warning("Analyst Node: No KEY_FACTS found in report.")
        except Exception as e:
            logger.error(f"Analyst Node: Error parsing KEY_FACTS: {e}")
        
        return {"report": report, "risk_score": risk_score, "revisions": revisions, "key_facts": key_facts}
        
    except Exception as e:
        logger.error(f"Error in analyst_node: {e}", exc_info=True)
        raise
