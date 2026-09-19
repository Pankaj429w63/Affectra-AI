import logging
import time
from typing import List

from backend.app.agents.context import AgentContext
from backend.app.agents.interpretation import InterpretationAgent
from backend.app.agents.knowledge import KnowledgeAgent
from backend.app.agents.response import ResponseAgent
from backend.app.agents.safety import SafetyAgent
from backend.app.agents.base import BaseAgent
from backend.app.llm.provider import LLMProvider

class AffectraAgentOrchestrator:
    """
    Orchestrates the execution of Affectra AI agents in a strictly controlled pipeline.
    """
    
    def __init__(self, provider: LLMProvider = None):
        """
        Initialize the orchestrator and instantiate the required agents.
        """
        self.interpretation_agent = InterpretationAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.response_agent = ResponseAgent(provider=provider)
        self.safety_agent = SafetyAgent()
        
        # Define strict execution order
        self.pipeline: List[BaseAgent] = [
            self.interpretation_agent,
            self.knowledge_agent,
            self.response_agent,
            self.safety_agent
        ]
        
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Executes the agent pipeline sequentially.
        """
        context.execution_trace = []
        
        for agent in self.pipeline:
            try:
                start_time = time.time()
                context = agent.execute(context)
                elapsed = time.time() - start_time
                
                # Record successful execution in trace
                context.execution_trace.append(f"{agent.name} (success, {elapsed:.3f}s)")
                
                # If an agent explicitly blocks/fails safely via context state, we continue to allow
                # safety agent to append disclaimers, etc. But if safety_approved is False, it's 
                # already handled by SafetyAgent. 
                
            except Exception as e:
                logging.error(f"Agent Pipeline failed at {agent.name}: {str(e)}")
                # Provide a clean, safe error message without leaking stack traces
                context.error_message = f"An internal error occurred during {agent.name} execution."
                context.final_response = "We encountered an unexpected error while generating the response."
                context.execution_trace.append(f"{agent.name} (FAILED)")
                context.safety_approved = False
                # Halt pipeline on unhandled exceptions
                break
                
        return context
