'''
                Shared Memory
              ┌──────────────┐
              │ Root Cause   │
              │ Fix Plan     │
              │ Project Rule │
              └──────────────┘
                    ▲
      ┌─────────────┼─────────────┐
      │             │             │
   Agent1       Agent2       Agent3
'''


import os

from crewai import Agent, Crew, LLM, Memory, Process, Task
from dotenv import load_dotenv


load_dotenv()

openrouter_llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# One memory for the entire crew.
shared_memory = Memory(
    llm=openrouter_llm,
    embedder={
        "provider": "sentence-transformer",
        "config": {
            "model": "all-MiniLM-L6-v2"
            }
        }
    )

# Add existing project information.
shared_memory.remember(
    content="The application uses Python 3.11 and FastAPI.",
    scope="/project/technology",
    categories=["project", "technology"],
    importance=0.9,
)

shared_memory.remember(
    content="API errors must be returned as JSON with code and message fields.",
    scope="/project/api-rules",
    categories=["project", "api", "rule"],
    importance=1.0,
)


bug_examiner = Agent(
    role="Bug Examiner",
    goal="Determine the cause of software bugs",
    backstory="Experienced in examining application errors.",
    llm=openrouter_llm,
    # No memory argument.
)

fix_planner = Agent(
    role="Fix Planner",
    goal="Create safe solutions for software bugs",
    backstory="Experienced in planning software fixes.",
    llm=openrouter_llm,
    # No memory argument.
)

test_planner = Agent(
    role="Test Planner",
    goal="Create tests that verify software fixes",
    backstory="Experienced in software testing.",
    llm=openrouter_llm,
    # No memory argument.
)


examine_task = Task(
    description=(
        "Examine this bug: {bug}. "
        "Use remembered project technology and API rules."
    ),
    expected_output="The probable cause of the bug.",
    agent=bug_examiner,
)

fix_task = Task(
    description=(
        "Create a fix for the diagnosed bug. "
        "Follow all remembered project rules."
    ),
    expected_output="A short fix plan.",
    agent=fix_planner
)

test_task = Task(
    description="Create three tests for the proposed fix.",
    expected_output="Three test cases with expected results.",
    agent=test_planner
)


crew = Crew(
    agents=[bug_examiner, fix_planner, test_planner],
    tasks=[examine_task, fix_task, test_task],
    process=Process.sequential,
    memory=shared_memory,
    verbose=True,
)


result = crew.kickoff(
    inputs={
        "bug": (
            "The login endpoint returns a plain text error when the password "
            "is incorrect."
        )
    }
)

print("\nFinal result:")
print(result)