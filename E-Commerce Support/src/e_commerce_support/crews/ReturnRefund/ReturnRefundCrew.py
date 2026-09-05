from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, task, crew
from e_commerce_support.tools.FetchOrderDetails import FetchOrderDetails
from e_commerce_support.KnowledgeSource.PolicyKnowledgeSource import LoadKnowledge
from dotenv import load_dotenv
load_dotenv("/workspaces/bles/.env")


def _build_llm() -> LLM:
    return LLM(
        model="openrouter/openai/gpt-4o-mini",
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )

@CrewBase
class ReturnRefundCrew:

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def request_parser(self) -> Agent:
        return Agent(
            config = self.agents_config["request_parser"],     #type: ignore
            llm = _build_llm()
        )
    @agent
    def order_verifier(self) -> Agent:
        return Agent(
            config = self.agents_config["order_verifier"],     #type: ignore
            llm = _build_llm(),
            tools = [FetchOrderDetails()]
        )

    @agent
    def policy_checker(self) -> Agent:
        return Agent(
            config = self.agents_config["policy_checker"],     #type: ignore
            llm = _build_llm(),
            knowledge_source = [LoadKnowledge()]
        )

    @agent
    def decision_maker(self) -> Agent:
        return Agent(
            config = self.agents_config["decision_maker"],     #type: ignore
            llm = _build_llm()
        )

    @agent
    def refund_processor(self) -> Agent:
        return Agent(
            config = self.agents_config["refund_processor"],     #type: ignore
            llm = _build_llm()
        )

    @agent
    def customer_communicator(self) -> Agent:
        return Agent(
            config = self.agents_config["customer_communicator"],     #type: ignore
            llm = _build_llm()
        )

    @task
    def parse_request_task(self) -> Task:
        return Task(
            config = self.tasks_config["parse_request_task"]      #type: ignore 
        )

    @task
    def verify_order_task(self) -> Task:
        return Task(
            config = self.tasks_config["verify_order_task"]      #type: ignore 
        )

    @task
    def check_policy_task(self) -> Task:
        return Task(
            config = self.tasks_config["check_policy_task"]      #type: ignore 
        )

    @task
    def decide_refund_task(self) -> Task:
        return Task(
            config = self.tasks_config["decide_refund_task"]      #type: ignore 
        )

    @task
    def process_refund_task(self) -> Task:
        return Task(
            config = self.tasks_config["process_refund_task"]      #type: ignore 
        )

    @task
    def notify_customer_task(self) -> Task:
        return Task(
            config = self.tasks_config["notify_customer_task"]      #type: ignore 
        )

    @crew
    def return_refund_crew(self) -> Crew:
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            process = Process.sequential,
            verbose = True
        )
