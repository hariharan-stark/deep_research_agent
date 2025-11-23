from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.researcher import researcher_node
from src.agents.analyst import analyst_node
from src.agents.reviewer import reviewer_node
from src.tools.search import get_google_search_tool
from langchain_core.messages import ToolMessage
from src.utils.logger import logger
import json

from langchain_core.runnables import RunnableConfig

def tools_node(state: AgentState, config: RunnableConfig):
    try:
        messages = state["messages"]
        last_message = messages[-1]
        
        tool = get_google_search_tool()
        
        results = []
        new_findings = []
        
        logger.info(f"Tools Node: Processing {len(last_message.tool_calls)} tool calls")
        
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "google_search":
                query = tool_call["args"]
                logger.info(f"Executing Google Search: {query}")
                try:
                    output = tool.invoke(query, config=config)
                    results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(output)))
                    new_findings.append(f"Query: {query}\nResult: {output}")
                except Exception as tool_err:
                    logger.error(f"Tool execution failed for {query}: {tool_err}")
                    results.append(ToolMessage(tool_call_id=tool_call["id"], content=f"Error: {str(tool_err)}"))
                
        return {"messages": results, "findings": new_findings}
        
    except Exception as e:
        logger.error(f"Error in tools_node: {e}", exc_info=True)
        raise

def create_graph():
    workflow = StateGraph(AgentState)
    

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reviewer", reviewer_node)
    
    workflow.set_entry_point("researcher")
    
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "tools"
            
        if state["iterations"] > 3:
            return "analyst"
            
        return "analyst"
        
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "tools": "tools",
            "analyst": "analyst"
        }
    )
    
    workflow.add_edge("tools", "researcher")
    
    def reviewer_decision(state: AgentState):
        feedback = state.get("reviewer_feedback", "")
        revisions = state.get("revisions", 0)
        
        if "DECISION: APPROVE" in feedback:
            return END
        
        # If rejected, check revision limit
        if revisions >= 2:
            logger.warning("Reviewer: Max revisions reached. Forcing END.")
            return END
            
        return "analyst"

    workflow.add_edge("analyst", "reviewer")
    
    workflow.add_conditional_edges(
        "reviewer",
        reviewer_decision,
        {
            END: END,
            "analyst": "analyst"
        }
    )
    
    return workflow.compile()
