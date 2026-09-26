import os

from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model = "openai/gpt-4o-mini",
    api_key = os.environ["OPENROUTER_API_KEY"],
    base_url = "https://openrouter.ai/api/v1",
    temperature = 0.7
)

class Blog(BaseModel):
    title: str
    content: str


blog_agent = Agent(
    role="Blog Content Generator Agent",
    goal="Generate a blog title and content",
    backstory="""You are an expert content creator, skilled in crafting engaging and informative blog posts.""",
    verbose=False,
    allow_delegation=False,
    llm=llm,
)

task1 = Task(
    description="""Create a blog title and content on a given topic. Make sure the content is under 200 words.""",
    expected_output="A compelling blog title and well-written content.",
    agent=blog_agent,
    output_pydantic=Blog,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[blog_agent],
    tasks=[task1],
    verbose=True,
    process=Process.sequential,
)

result = crew.kickoff()
print(result)

# Option 1: Accessing Properties Using Dictionary-Style Indexing
print("Accessing Properties - Option 1")
title = result["title"]         #type:ignore
content = result["content"]         #type:ignore
print("Title:", title)
print("Content:", content)

# Option 2: Accessing Properties Directly from the Pydantic Model
print("Accessing Properties - Option 2")
title = result.pydantic.title          #type:ignore
content = result.pydantic.content         #type:ignore
print("Title:", title)
print("Content:", content)

# Option 3: Accessing Properties Using the to_dict() Method
print("Accessing Properties - Option 3")
output_dict = result.to_dict()         #type:ignore
title = output_dict["title"]
content = output_dict["content"]
print("Title:", title)
print("Content:", content)

# Option 4: Printing the Entire Blog Object
print("Accessing Properties - Option 4")
print("Blog:", result)

"""
To use it in JSON format output 

# Define the task with output_json set to the Blog model
task1 = Task(
    description="Create a blog title and content on a given topic. Make sure the content is under 200 words.",
    expected_output="A JSON object with 'title' and 'content' fields.",
    agent=blog_agent,
    output_json=Blog,
)
"""