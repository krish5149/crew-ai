from crewai.flow.flow import Flow, start, listen, or_
from crewai.flow.human_feedback import human_feedback
from pydantic import BaseModel
from crewai.flow.human_feedback import HumanFeedbackResult
from crewai import LLM
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

'''
Parameters in HumanFeedbackResult dataclass
    output: Any
    feedback: str
    outcome: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    method_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
'''


open_router = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
) 

class ContentState(BaseModel):
    draft: str = ""
    revision_count: int = 0
    status: str = "pending"

class ContentApprovalWorkFlow(Flow[ContentState]):

    @start()
    def generate_draft(self):
        self.state.draft = '# AI Safety\n\nThis is a draft about AI Safety...'
        return self.state.draft

    @human_feedback(
            message = "Please Review this draft, Approve, Reject or describe what it needs...",
            emit = ["approved","rejected","need_revision"],
            llm = open_router,
            default_outcome = "need_revision",
            metadata = {"Reviewer":"Ramki"}
    )
    @listen(or_(generate_draft,"need_revision"))
    def process_feedback(self):
        self.state.revision_count += 1
        return f"{self.state.draft} ({self.state.revision_count})"

    @listen("approved")
    def publish_content(self, result: HumanFeedbackResult):
        self.state.status = "Published"
        print(result.feedback)
        print(result.method_name)
        print(result.metadata)

    @listen("rejected")
    def rejected_content(self, result: HumanFeedbackResult):
        self.state.status = "Rejected"
        print(result.feedback)
        print(result.method_name)
        print(result.metadata)
        


flow = ContentApprovalWorkFlow()
flow.plot("ContentApprovalWorkFlowHITL")
flow.kickoff()