from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, task, crew

from dotenv import load_dotenv
load_dotenv("/workspaces/bles/.env")

from application_assistant.tools.PDFExporterTool import PDFExporterTool
from application_assistant.tools.ResumeReaderTool import ResumeReaderTool
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

def _build_llm() -> LLM:
    return LLM(
        model="openrouter/openai/gpt-4o-mini",
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )


@CrewBase
class ResumeGenerateCrew:

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def job_researcher(self) -> Agent:
        return Agent(
            config = self.agents_config["job_researcher"],        # type: ignore[index]
            llm = _build_llm(),
            tools = [SerperDevTool(), ScrapeWebsiteTool()]
        )

    @agent
    def job_matcher(self) -> Agent:
        return Agent(
            config = self.agents_config["job_matcher"],        # type: ignore[index]
            llm = _build_llm(),
            tools = [ResumeReaderTool()]
        )

    @agent
    def resume_writer(self) -> Agent:
        return Agent(
            config = self.agents_config["job_matcher"],        # type: ignore[index]
            llm = _build_llm(),
            tools = [PDFExporterTool(), ResumeReaderTool()]
        )

    @agent
    def cover_letter_writer(self) -> Agent:
        return Agent(
            config = self.agents_config["cover_letter_writer"],        # type: ignore[index]
            llm = _build_llm(),
            tools = [ResumeReaderTool()]
        )

    @agent
    def application_reviewer(self) -> Agent:
        return Agent(
            config = self.agents_config["cover_letter_writer"],        # type: ignore[index]
            llm = _build_llm(),
            tools = [PDFExporterTool()]
        )

    @task
    def research_jobs_task(self) -> Task:
        return Task(
            config = self.tasks_config["research_jobs_task"]           #type: ignore
        )

    @task
    def match_jobs_task(self) -> Task:
        return Task(
            config = self.tasks_config["match_jobs_task"]           #type: ignore
        )

    @task
    def customize_resumes_task(self) -> Task:
        return Task(
            config = self.tasks_config["customize_resumes_task"]           #type: ignore
        )

    @task
    def write_cover_letters_task(self) -> Task:
        return Task(
            config = self.tasks_config["write_cover_letters_task"]           #type: ignore
        )

    @task
    def review_applications_task(self) -> Task:
        return Task(
            config = self.tasks_config["review_applications_task"]           #type: ignore
        )

    @crew
    def crew(self):
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            process = Process.sequential,
            verbose = True
        )

    