import chainlit as cl
import sys
import os

# Add project root to sys.path (optional if running from root, but safe to keep)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.graph import create_graph
from src.config import config
from src.utils.logger import logger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
import json

MODEL_OPTIONS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gpt-4", "gpt-4o"]

@cl.on_chat_start
async def start():
    logger.info("Chat session started")

    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="researcher_model",
                label="Researcher Model",
                values=MODEL_OPTIONS,
                initial_index=0,
            ),
            cl.input_widget.Select(
                id="analyst_model",
                label="Analyst Model",
                values=MODEL_OPTIONS,
                initial_index=0,
            ),
        ]
    ).send()
    
    await cl.Message(
        content="### Welcome to the Deep Research Agent! 🕵️‍♂️\n\nI am your concierge. To get started, please tell me the **name of the person or entity** you would like to investigate.",
        author="Concierge"
    ).send()
    
    cl.user_session.set("settings", settings)

@cl.on_settings_update
async def setup_agent(settings):
    logger.info(f"Settings updated: {settings}")
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received message: {message.content}")
    # Retrieve settings
    settings = cl.user_session.get("settings")
    researcher_model = settings["researcher_model"]
    analyst_model = settings["analyst_model"]
    
    # --- Concierge Logic ---
    # Use the researcher model for the concierge (lightweight check)
    if researcher_model.startswith("gpt"):
        concierge_model = ChatOpenAI(model=researcher_model, temperature=0)
    else:
        concierge_model = ChatVertexAI(model_name=researcher_model, temperature=0)
    
    concierge_system_prompt = """You are a helpful Concierge for a Deep Research Agent.
    Your goal is to determine if the user is asking for research or just chatting.

    1. If the user sends a greeting (Hi, Hello) or asks general questions (What can you do?), reply conversationally and politely. Ask them who they want to research.
    2. If the user provides a name, entity, or topic to research, OR asks to research something, output EXACTLY: "RESEARCH_REQUEST: <topic>"

    Examples:
    User: "Hi"
    Output: "Hello! I'm ready to help you research. Who would you like to research today?"

    User: "Elon Musk"
    Output: "RESEARCH_REQUEST: Elon Musk"

    User: "Can you research Apple Inc?"
    Output: "RESEARCH_REQUEST: Apple Inc."

    User: "What is the weather?"
    Output: "I am a specialized Research Agent. I can't check the weather, but I can research people or companies for you."
    """

    messages = [
        SystemMessage(content=concierge_system_prompt),
        HumanMessage(content=message.content)
    ]
    
    response = await concierge_model.ainvoke(messages)
    content = response.content.strip()
    
    if "RESEARCH_REQUEST:" in content:
        topic = content.split("RESEARCH_REQUEST:")[1].strip()
        logger.info(f"Starting research on topic: {topic}")
        
        await cl.Message(
            content=f"Starting deep research on **{topic}** using:\n- Researcher: `{researcher_model}`\n- Analyst: `{analyst_model}`\n\nThis may take a few minutes...",
            author="Concierge"
        ).send()
        
        graph = create_graph()
        
        initial_state = {
            "research_topic": topic,
            "findings": [],
            "search_queries": [],
            "iterations": 0,
            "risk_score": 0.0,
            "report": "",
            "messages": [],
            "researcher_model": researcher_model,
            "analyst_model": analyst_model,
            "revisions": 0
        }
        
        # Create a parent step to group all actions
        async with cl.Step(name="Deep Research Process", type="run") as parent_step:
            parent_step.input = f"Starting research on: {topic}"
            
            final_report = ""
            final_risk_score = "N/A"

            # Stream events
            async for event in graph.astream(initial_state):
                for key, value in event.items():
                    if key == "researcher":
                        async with cl.Step(name="Researcher", type="llm", parent_id=parent_step.id) as step:
                            step.input = "Analyzing current findings and deciding next steps..."
                            
                            if "messages" in value and value["messages"]:
                                last_msg = value["messages"][-1]
                                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                    queries = [tc['args'].get('query', 'Unknown') for tc in last_msg.tool_calls]
                                    step.output = f"Decided to search for: {', '.join(queries)}"
                                else:
                                    step.output = "Sufficient information gathered. Handing off to Analyst."
                            else:
                                step.output = "Processing..."

                    elif key == "tools":
                        async with cl.Step(name="Google Search", type="tool", parent_id=parent_step.id) as step:
                            step.input = "Executing search queries..."
                            if "findings" in value:
                                new_findings = value["findings"]
                                formatted_findings = "\n\n".join(new_findings)
                                step.output = formatted_findings
                            else:
                                step.output = "No new findings."
                    
                    elif key == "analyst":
                        if "report" in value:
                            final_report = value["report"]
                            final_risk_score = value.get("risk_score", "N/A")
                            
                            async with cl.Step(name="Analyst", type="llm", parent_id=parent_step.id) as step:
                                step.input = "Drafting intelligence report..."
                                step.output = "Report drafted. Sending to Reviewer."
                    
                    elif key == "reviewer":
                        async with cl.Step(name="Reviewer", type="llm", parent_id=parent_step.id) as step:
                            step.input = "Checking report for accuracy and consistency..."
                            if "reviewer_feedback" in value:
                                feedback = value["reviewer_feedback"]
                                step.output = f"Feedback: {feedback}"
                            else:
                                step.output = "Review complete."
            
            parent_step.output = "Research completed."
            
            if final_report:
                await cl.Message(
                    content=f"## 🚨 Final Risk Assessment\n\n**Risk Score**: {final_risk_score}/10\n\n{final_report}",
                    author="Analyst"
                ).send()

                # Save the report to a file
                try:
                    safe_topic = "".join([c for c in topic if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                    safe_topic = safe_topic.replace(' ', '_')
                    
                    report_dir = os.path.join("reports", safe_topic)
                    os.makedirs(report_dir, exist_ok=True)
                    
                    file_path = os.path.join(report_dir, "risk_assessment.md")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(f"# Risk Assessment: {topic}\n\n")
                        f.write(f"**Date**: {os.path.abspath(file_path)}\n") # Placeholder for date or just path
                        f.write(f"**Risk Score**: {final_risk_score}/10\n\n")
                        f.write(final_report)
                    
                    logger.info(f"Report saved to {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save report: {str(e)}")
    else:
        await cl.Message(content=content, author="Concierge").send()
