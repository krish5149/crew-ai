import os
from typing import Any, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, after_kickoff, agent, before_kickoff, crew, task

from demo.tools.custom_tool import WordCountTool
from dotenv import load_dotenv

load_dotenv()
def _build_llm() -> LLM:
    return LLM(
        model="openrouter/openai/gpt-4o-mini",
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )

def _research_tools() -> List[Any]:
    """Search tool is opt-in: without SERPER_API_KEY the crew still runs."""
    tools: List[Any] = [WordCountTool()]
    if os.getenv("SERPER_API_KEY"):
        from crewai_tools import SerperDevTool

        tools.append(SerperDevTool())
    return tools


@CrewBase
class ResearchCrew:
    """A two-agent research-and-report crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ------------------------------------------------------------------ hooks

    @before_kickoff
    def prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalise inputs before the first agent runs."""
        if not inputs.get("topic"):
            raise ValueError("`topic` is required in the kickoff inputs")
        os.makedirs("output", exist_ok=True)
        return inputs

    @after_kickoff
    def log_result(self, result):
        """Hook for persistence, metrics, or notification."""
        usage = getattr(result, "token_usage", None)
        if usage is not None:
            print(f"[research_crew] token usage: {usage}")
        return result

    # ----------------------------------------------------------------- agents

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],        #type: ignore
            llm=_build_llm(),
            tools=_research_tools(),
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_analyst"],        #type: ignore
            llm=_build_llm(),
        )

    # ------------------------------------------------------------------ tasks

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])        #type: ignore

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"])        #type: ignore

    # ------------------------------------------------------------------- crew

    @crew
    def crew(self) -> Crew:
        """The @agent and @task decorators populate self.agents / self.tasks."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
        )
