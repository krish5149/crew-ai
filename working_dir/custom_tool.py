# Creating pandas DB for tool

import pandas as pd
import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, Process, LLM

load_dotenv()

data = {
    "customer_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    "relationship": [
        "Gold", "Silver", "Platinum", "Bronze", "Gold",
        "Silver", "Platinum", "Bronze", "Gold", "Silver"
    ],
    "status_for_upgrade": [
        "Eligible", "Not Eligible", "Eligible", "Not Eligible", "Eligible",
        "Not Eligible", "Eligible", "Not Eligible", "Eligible", "Not Eligible"
    ]
}

df = pd.DataFrame(data) 
customers_ids = df["customer_id"]

llm = LLM(
    model = "openai/gpt-4o-mini",
    api_key = os.environ["OPENROUTER_API_KEY"],
    base_url = "https://openrouter.ai/api/v1",
    temperature = 0.7
)

# define the tool input schema

from typing import Type
from pydantic import BaseModel, Field

class DBSearchToolInputScema(BaseModel):

    customerid: int = Field(
        description = "customer id to return that the user is eligible or not"
    )

# define the task ouput schema

class CustomerEligibilityOutputScema(BaseModel):
    status: str
    customerid: int

# Create a custom tool

from crewai.tools import BaseTool

class DBSearchTool(BaseTool):

    name: str = "Customer DB Query"
    description: str = "Checks the Eligibility of the customer"

    args_schema: Type[BaseModel] = DBSearchToolInputScema

    def _run(self, customerid: int) -> dict: 

        try:
            if customerid in customers_ids.values:
                print(f" The customer Id {customerid} found")
                result = df.loc[df["customer_id"]==customerid,"status_for_upgrade"]
                status = result.iloc[0]
                return {
                    "status" : status
                }
            else:
                return {
                    "status" : "Not Eligible"
                }
        except Exception as e:
            return {"error": "Check the customer id"}


# assigning tool

EligibilityCheckerTool = DBSearchTool()

# creating agent

eligibility_agent = Agent(
    role="Customer Eligibility Verification Specialist",
    goal="Determine whether the customer is Eligible or Not Eligible using the eligibility tool.",
    backstory=(
        "You are an automated assessment agent responsible for taking {customerid}"
        "and running it through the EligibilityCheckerTool tool to get a definitive eligibility status."
    ),
    llm=llm,
    tools=[EligibilityCheckerTool],
    allow_delegation=False,
    verbose=True,
)

next_action_agent = Agent(
    role="Action Coordinator",
    goal="Process structured eligibility results and execute the correct onboarding or rejection workflow.",
    backstory="You parse structured evaluation data to draft formal communication and outline required next steps.",
    llm=llm,
    verbose=True
)

# creating task

eligibility_task = Task(
    description=(
        "Check eligibility for the customer with customerid: {customerid}"
        "Use the Eligibility Checker Tool to get the explicit status."
    ),
    expected_output="An instance of EligibilityResult with status, customerid",
    agent=eligibility_agent,
    output_pydantic=CustomerEligibilityOutputScema
)

task2_action = Task(
    description=(
        "Review the structured evaluation data from Task 1. "
        "Check the 'status' field: "
        "- If Eligible: Draft an acceptance email and onboarding checklist. "
        "- If Not Eligible: Draft a courteous rejection letter citing the 'reason' field."
    ),
    expected_output="A formal email draft with appropriate next steps based on the structured decision.",
    agent=next_action_agent,
    context=[eligibility_task]  # Passes structured Pydantic output into Agent 2 context
)

# creating crew 

crew = Crew(
    agents=[eligibility_agent, next_action_agent],
    tasks=[eligibility_task, task2_action],
    process=Process.sequential,
    verbose=True,
)

# running crew

result = crew.kickoff(inputs={"customerid":1001})

print(result)


