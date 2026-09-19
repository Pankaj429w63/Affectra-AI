import logging
from fastapi import APIRouter, HTTPException
from backend.app.schemas.agents import AgentAPIRequest, AgentAPIResponse, AgentContextItem
from backend.app.agents.context import AgentContext
from backend.app.agents.orchestrator import AffectraAgentOrchestrator

router = APIRouter()

# Instantiate the orchestrator once per worker to reuse the retriever and model cache
# Dependencies like AffectraRetriever will be lazily loaded by KnowledgeAgent within the orchestrator
_orchestrator = AffectraAgentOrchestrator()

@router.post("/agents", response_model=AgentAPIResponse, summary="Agent Pipeline")
def run_agent_pipeline(request: AgentAPIRequest):
    """
    Executes the full Affectra AI Agent pipeline.
    
    1. InterpretationAgent
    2. KnowledgeAgent
    3. ResponseAgent
    4. SafetyAgent
    """
    
    # Optional strict validation: you can reject empty query if required, but the schema allows it.
    if request.user_query is not None and not request.user_query.strip():
        raise HTTPException(status_code=422, detail="user_query cannot be empty whitespace.")
        
    try:
        # Build the initial agent context from the API request
        context = AgentContext(
            user_query=request.user_query,
            emotion_label=request.emotion_label,
            emotion_probabilities=request.emotion_probabilities,
            sentiment_label=request.sentiment_label,
            sentiment_probabilities=request.sentiment_probabilities,
            top_k=request.top_k
        )
        
        # Execute the orchestrator
        final_context = _orchestrator.execute(context)
        
        # Determine appropriate HTTP status based on errors
        if final_context.error_message:
            # We don't throw an HTTPException here because the orchestrator gracefully handles 
            # failures and returns a safe state, but if we wanted to 500 we could.
            # The prompt asks us to return structured failure or appropriate status.
            # Returning a 500 is standard for internal agent failure.
            raise HTTPException(status_code=500, detail=final_context.error_message)
            
        # Parse retrieved context
        retrieved_items = []
        if final_context.retrieved_context_items:
            retrieved_items = [
                AgentContextItem(**item) for item in final_context.retrieved_context_items
            ]
            
        # Construct the API response
        response = AgentAPIResponse(
            final_response=final_context.final_response,
            safety_approved=final_context.safety_approved,
            execution_trace=final_context.execution_trace,
            retrieved_context=retrieved_items
        )
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"API Route Error: {e}")
        # Safe catch-all that doesn't leak stack traces
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing the agent pipeline.")
