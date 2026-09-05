from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, task, crew
from e_commerce_support.tools.CreateSupportTicket import CreateSupportTicket
import os
from e_commerce_support.crews.EsclationManager.models.EsclationStructuredOutput import EscalationStructuredOutput  #type: ignore
from dotenv import load_dotenv
load_dotenv("/workspaces/bles/.env")


def _build_llm() -> LLM:
    return LLM(
        model="openrouter/openai/gpt-4o-mini",
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )

@CrewBase
class EsclationManagerCrew:

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def escalation_agent(self) -> Agent:
        return Agent(
            config = self.agents_config["escalation_agent"],     #type: ignore
            llm = _build_llm(),
            tools = [CreateSupportTicket()]
        )

    @task
    def escalation_task(self) -> Task:
        return Task(
            config = self.tasks_config["escalation_task"],      #type: ignore  
            output_pydantic = EscalationStructuredOutput,
        )

    @crew
    def esclation_crew(self) -> Crew:
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            process = Process.sequential,
            verbose = True
        )
