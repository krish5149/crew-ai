import os
from crewai import Memory, LLM, Agent, Task, Crew, Process
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to the .env file.")

openrouter_llm = LLM(
    model = "openai/gpt-4o-mini",
    api_key = os.environ["OPENROUTER_API_KEY"],
    base_url = "https://openrouter.ai/api/v1",
    temperature = 0.7
)

technical_memory = Memory(
    llm=openrouter_llm,
    embedder={
        "provider": "sentence-transformer",
        "config": {
            "model": "all-MiniLM-L6-v2"
            }
        }
    )

class IncidentClassification(BaseModel):
    scope: Literal["constraint", "server", "memory"] = Field(
        description="The memory scope most relevant to the incident."
    )
    reason: str = Field(
        description="A short explanation for selecting this scope."
    )
  
def seed_memory() -> None:

    technical_memory.remember(
        content=(
            "The inventory API must not be restarted while an inventory "
            "synchronization job is active."
        ),
        scope="/knowledge/constraint/inventory-api",
        categories=["constraint", "inventory-api", "restart"],
        importance=1.0,
        source="operations-policy",
    )

    technical_memory.remember(
        content=(
            "The inventory API runs on APP-01, listens on port 8080, "
            "and is managed by the inventory-api service."
        ),
        scope="/knowledge/server/app-01",
        categories=["server", "configuration", "inventory-api"],
        importance=0.9,
        source="server-documentation",
    )

    technical_memory.remember(
        content=(
            "APP-01 has 4 GB of RAM. The inventory API normally uses "
            "less than 1.5 GB."
        ),
        scope="/knowledge/memory/app-01",
        categories=["memory", "server", "inventory-api"],
        importance=0.9,
        source="server-documentation",
    )

    technical_memory.remember(
        content=(
            "A previous inventory API memory leak was resolved by stopping "
            "the synchronization worker and restarting the service."
        ),
        scope="/knowledge/memory/resolutions",
        categories=["memory", "verified-resolution", "inventory-api"],
        importance=0.95,
        source="incident-history",
    )

# Incident Router Agent
incident_router = Agent(
    role="Incident Router",
    goal=(
        "Classify each incident into exactly one technical memory scope: "
        "constraint, server, or memory."
    ),
    backstory=(
        "You quickly route incidents to the most relevant technical "
        "knowledge branch."
    ),
    llm=openrouter_llm,
    verbose=True,
)

# Incident Router Task
route_task = Task(
    description=(
        "Classify this incident into exactly one scope:\n"
        "- constraint: policy, permission, maintenance window, or restriction\n"
        "- server: connectivity, service, port, host, disk, or configuration\n"
        "- memory: RAM usage, memory leak, OOM, or process termination\n\n"
        "Server: {server}\n"
        "Service: {service}\n"
        "Symptoms: {symptoms}\n"
        "Logs: {logs}"
    ),
    expected_output=(
        "A structured classification containing the selected scope and reason."
    ),
    agent=incident_router,
    output_pydantic=IncidentClassification,
)

routing_crew = Crew(
    agents=[incident_router],
    tasks=[route_task],
    memory=False,
    verbose=True,
)

def classify_incident(incident_inputs: dict[str, str]) -> IncidentClassification:
    result = routing_crew.kickoff(inputs=incident_inputs)
    classification = result.pydantic   #type: ignore

    if classification is None:
        raise RuntimeError("The router did not return a valid classification.")

    return classification   #type: ignore


def create_maintenance_crew(selected_scope: str) -> Crew:
    scoped_memory = technical_memory.scope(
        f"/knowledge/{selected_scope}"
    )

    system_monitor = Agent(
        role="System Monitor",
        goal=(
            "Diagnose the current incident using only the selected relevant "
            "technical memory branch."
        ),
        backstory=(
            "You specialize in server logs, application failures, and "
            "production diagnosis."
        ),
        llm=openrouter_llm,
        memory=scoped_memory,
        verbose=True,
    )

    repair_engineer = Agent(
        role="Repair Engineer",
        goal=(
            "Create a safe repair plan using the diagnosis and relevant "
            "technical memory."
        ),
        backstory=(
            "You specialize in reversible repairs, validation, and rollback."
        ),
        llm=openrouter_llm,
        memory=scoped_memory,
        verbose=True,
    )

    incident_reporter = Agent(
        role="Incident Reporter",
        goal="Summarize the current diagnosis and repair plan.",
        backstory="You create concise technical incident reports.",
        llm=openrouter_llm,
        # No persistent memory.
        verbose=True,
    )

    monitor_task = Task(
        description=(
            "Analyze this incident:\n\n"
            "Selected memory scope: {selected_scope}\n"
            "Server: {server}\n"
            "Service: {service}\n"
            "Symptoms: {symptoms}\n"
            "Logs: {logs}\n\n"
            "Use relevant information from your assigned memory scope. "
            "Determine the probable cause and supporting evidence."
        ),
        expected_output=(
            "A probable cause, supporting evidence, and confidence level."
        ),
        agent=system_monitor,
    )

    repair_task = Task(
        description=(
            "Create a safe repair procedure using the diagnosis and your "
            "assigned memory scope. Include validation and rollback steps."
        ),
        expected_output=(
            "A numbered repair procedure with validation and rollback."
        ),
        agent=repair_engineer,
        context=[monitor_task],
    )

    report_task = Task(
        description=(
            "Summarize the diagnosis and repair procedure from the current "
            "tasks. Do not add information that is not present in the results."
        ),
        expected_output="A concise incident report.",
        agent=incident_reporter,
        context=[monitor_task, repair_task],
    )

    return Crew(
        agents=[
            system_monitor,
            repair_engineer,
            incident_reporter,
        ],
        tasks=[
            monitor_task,
            repair_task,
            report_task,
        ],
        process=Process.sequential,
        # Memory is shared only by the selected agents, not the whole crew.
        memory=False,
        verbose=True,
    )

def run_incident(incident_inputs: dict[str, str]) -> None:
    print("\n=== Stage 1: Classifying incident ===")

    classification = classify_incident(incident_inputs)

    print(f"\nSelected scope: {classification.scope}")
    print(f"Reason: {classification.reason}")

    scoped_inputs = {
        **incident_inputs,
        "selected_scope": classification.scope,
    }

    print("\n=== Stage 2: Diagnosing and repairing ===")

    maintenance_crew = create_maintenance_crew(
        classification.scope
    )

    result = maintenance_crew.kickoff(inputs=scoped_inputs)

    print("\n=== Final report ===")
    print(result)


def main() -> None:
    seed_memory()

    incident = {
        "server": "APP-01",
        "service": "inventory-api",
        "symptoms": (
            "The API becomes slow and eventually returns HTTP 503."
        ),
        "logs": (
            "Memory usage reached 3.8 GB. The operating system terminated "
            "the process after an out-of-memory condition."
        ),
    }

    run_incident(incident)


if __name__ == "__main__":
    main()


'''
Incident
   ↓
Incident Router
   ↓
Classification: memory
   ↓
technical_memory.scope("/knowledge/memory")
   ↓
System Monitor and Repair Engineer
   ↓
Incident Reporter receives only current task context
'''

"""
Incident
   |
   v
Router Agent
   |
   v
Classification = memory
   |
   v
technical_memory.scope("/knowledge/memory")
   |
   +--> APP-01 has 4 GB RAM
   +--> Previous memory leak resolution
   +--> Inventory API memory knowledge
   |
   v
System Monitor
   |
   v
Diagnosis: Memory Leak / OOM
   |
   v
Repair Engineer
   |
   v
Repair Steps + Validation + Rollback
   |
   v
Incident Reporter
   |
   v
Final Incident Report
"""