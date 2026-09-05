import random
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class CreateSupportTicketInput(BaseModel):
    """Input schema for CreateSupportTicket tool."""
    query: str = Field(..., description="The user's issue or query that needs a support ticket.")


class CreateSupportTicket(BaseTool):
    name: str = "CreateSupportTicket"
    description: str = "Creates a support ticket for the user's query and returns a ticket ID."
    args_schema: type[BaseModel] = CreateSupportTicketInput

    def _run(self, query: str) -> str:
        ticket_id = random.randint(10000, 99999)
        return f"Ticket ID: {ticket_id}, is generated for issue: '{query}'"