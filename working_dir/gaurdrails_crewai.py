"""
Task Guardrails
Task guardrails provide a way to validate and transform task outputs before they are passed to the next task. This feature helps ensure data quality and provides feedback to agents when their output doesn’t meet specific criteria.

CrewAI supports two types of guardrails:

Function-based guardrails: Python functions with custom validation logic, giving you complete control over the validation process and ensuring reliable, deterministic results.

LLM-based guardrails: String descriptions that use the agent’s LLM to validate outputs based on natural language criteria. These are ideal for complex or subjective validation requirements.
"""
from dotenv import load_dotenv
import os
from typing import Any
from crewai.tasks.task_output import TaskOutput
from crewai import Agent, Task, Crew, Process, LLM
load_dotenv()

llm = LLM(
    model = "openai/gpt-4o-mini",
    api_key = os.environ["OPENROUTER_API_KEY"],
    base_url = "https://openrouter.ai/api/v1",
    temperature = 0.7
)


# Function-based guardrails / Multiple Function-based guardrails:

def validate_answer(result: TaskOutput) -> tuple[bool, Any]:

    if len(result.raw.strip()) < 200000000:
        return (False, "Blog content exceeds 200000000 words")
    return (True,result.raw.strip())

def check_content(result: TaskOutput) -> tuple[bool, Any]:

    if result.raw.strip():
        return (True, "Good Content")
    return (True,result.raw.strip())

researcher = Agent(
    role="Software Testing Researcher",
    goal=(
        "Research reliable information about automated software testing "
        "and explain the findings clearly."
    ),
    backstory=(
        "You are an experienced software researcher who specializes in "
        "software quality, automated testing, and development practices. "
        "You verify facts and organize findings into clear explanations."
    ),
    llm = llm,
    verbose=True,
    allow_delegation=False,
)

research_task = Task(
    description="Explain the benefits of automated testing.{topic}",
    expected_output="A detailed explanation containing at least 50 characters.",
    agent=researcher,
    guardrail=[validate_answer,check_content] #type: ignore
)

crew = Crew(
    agents = [researcher],
    tasks = [research_task],
    process = Process.sequential,
    verbose = True
)

result = crew.kickoff(inputs={"topic":"what is automation"})


# LLM-based guardrails

researcher = Agent(
    role="Software Testing Researcher",
    goal=(
        "Research reliable information about automated software testing "
        "and explain the findings clearly."
    ),
    backstory=(
        "You are an experienced software researcher who specializes in "
        "software quality, automated testing, and development practices. "
        "You verify facts and organize findings into clear explanations."
    ),
    llm = llm,
    verbose=True,
    allow_delegation=False,
)

research_task = Task(
    description="Explain the benefits of automated testing.{topic}",
    expected_output="A detailed explanation containing at least 50 characters.",
    agent=researcher,
    guardrail="The blog post must contains the name Ram",
    guardrail_max_retries=2
)

crew = Crew(
    agents = [researcher],
    tasks = [research_task],
    process = Process.sequential,
    verbose = True
)

result = crew.kickoff(inputs={"topic":"what is automation"})


# Mix function-based and LLM-based guardrails

'''
blog_task = Task(
    description="Write a blog post about AI",
    expected_output="A well-formatted blog post between 100-500 words",
    agent=blog_agent,
    guardrails=[
        validate_word_count,  # Function-based: precise word count check

        "The content must be engaging and suitable for a general audience",  # LLM-based: subjective quality check

        "The writing style should be clear, concise, and free of technical jargon"  # LLM-based: style validation
    ],
    guardrail_max_retries=3
)'''
