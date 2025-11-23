from typing import Dict, Any, List, Optional, Union
from uuid import UUID
import chainlit as cl
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class CustomChainlitCallbackHandler(BaseCallbackHandler):
    """
    A custom callback handler to stream LangChain events to Chainlit steps.
    This avoids the import errors found in the built-in handler with newer LangChain versions.
    """
    
    def __init__(self):
        self.steps: Dict[str, cl.Step] = {}
        
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        # We can filter which chains to show. For now, show all top-level or interesting ones.
        # In LangGraph, nodes are often chains.
        name = serialized.get("name", "Chain") if serialized else "Chain"
        
        # Create a step
        step = cl.Step(name=name, type="chain", parent_id=str(parent_run_id) if parent_run_id else None)
        step.input = str(inputs)
        self.steps[str(run_id)] = step
        cl.run_sync(step.send())

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        if str(run_id) in self.steps:
            step = self.steps[str(run_id)]
            step.output = str(outputs)
            cl.run_sync(step.update())
            self.steps.pop(str(run_id))

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        if str(run_id) in self.steps:
            step = self.steps[str(run_id)]
            step.is_error = True
            step.output = str(error)
            cl.run_sync(step.update())
            self.steps.pop(str(run_id))

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        step = cl.Step(name="LLM", type="llm", parent_id=str(parent_run_id) if parent_run_id else None)
        step.input = prompts[0] if prompts else ""
        self.steps[str(run_id)] = step
        cl.run_sync(step.send())

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        if str(run_id) in self.steps:
            step = self.steps[str(run_id)]
            step.output = response.generations[0][0].text
            cl.run_sync(step.update())
            self.steps.pop(str(run_id))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        name = serialized.get("name", "Tool") if serialized else "Tool"
        step = cl.Step(name=name, type="tool", parent_id=str(parent_run_id) if parent_run_id else None)
        step.input = input_str
        self.steps[str(run_id)] = step
        cl.run_sync(step.send())

    def on_tool_end(
        self, output: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        if str(run_id) in self.steps:
            step = self.steps[str(run_id)]
            step.output = str(output)
            cl.run_sync(step.update())
            self.steps.pop(str(run_id))

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        if str(run_id) in self.steps:
            step = self.steps[str(run_id)]
            step.is_error = True
            step.output = str(error)
            cl.run_sync(step.update())
            self.steps.pop(str(run_id))
