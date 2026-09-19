from abc import ABC, abstractmethod
from backend.app.agents.context import AgentContext

class BaseAgent(ABC):
    """
    Common abstraction for all Affectra AI agents.
    Every agent takes an AgentContext, modifies it, and returns the modified context.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        pass
        
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Execute the agent's specific responsibility.
        Args:
            context: The shared AgentContext
        Returns:
            The modified AgentContext
        """
        pass
