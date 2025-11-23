from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    research_topic: str
    findings: Annotated[List[str], operator.add]
    search_queries: List[str]

class AgentState(TypedDict):
    research_topic: str
    findings: Annotated[List[str], operator.add]
    search_queries: List[str]
    iterations: int
    risk_score: float
    report: str
    messages: Annotated[List[any], operator.add]
    researcher_model: str
    analyst_model: str
    key_facts: List[str] # Preserved critical facts
    reviewer_feedback: str # Feedback from Reviewer Agent
    revisions: int # Track number of review cycles
