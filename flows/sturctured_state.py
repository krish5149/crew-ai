from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel

class BlogState(BaseModel):
    title: str = ""
    word_count: int = 0
    status: str = ""

class BlogFlow(Flow[BlogState]):

    @start()
    def draft_post(self):
        self.state.title = "Introduction to Dr. DOOM"
        self.state.word_count = 500
        self.state.status = "Pend"

    @listen(draft_post)
    def review_post(self):
        self.state.word_count += 200
        self.state.status = "Reviewed"

    @listen(review_post)
    def publish_post(self):
        self.state.status = "Published"


flow = BlogFlow()
flow.plot("BlogState")

final_output = flow.kickoff()
print(flow.state)