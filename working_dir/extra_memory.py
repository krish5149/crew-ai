import argparse
import os
from typing import Any

from crewai import Agent, Crew, LLM, Memory, Process, Task
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. Add it to the .env file."
    )

openrouter_llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


# Demonstrates semantic, recency, importance, and recency-decay configuration.
memory = Memory(
    llm=openrouter_llm,
    semantic_weight=0.5,
    recency_weight=0.3,
    importance_weight=0.2,
    recency_half_life_days=14,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def heading(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def show_record(record: Any) -> None:
    """Safely display a MemoryRecord."""

    if record is None:
        print("No record was created, possibly because it was a duplicate.")
        return

    print(f"ID: {record.id}")
    print(f"Content: {record.content}")
    print(f"Scope: {record.scope}")
    print(f"Categories: {record.categories}")
    print(f"Metadata: {record.metadata}")
    print(f"Importance: {record.importance}")
    print(f"Created: {record.created_at}")
    print(f"Last accessed: {record.last_accessed}")
    print(f"Source: {record.source}")
    print(f"Private: {record.private}")
    print(f"Has embedding: {record.embedding is not None}")


def show_matches(matches: list[Any]) -> None:
    """Display MemoryMatch objects."""

    if not matches:
        print("No relevant memories found.")
        return

    for number, match in enumerate(matches, start=1):
        print(f"\nMatch {number}")
        print(f"Score: {match.score:.3f}")
        print(f"Content: {match.record.content}")
        print(f"Scope: {match.record.scope}")
        print(f"Categories: {match.record.categories}")
        print(f"Importance: {match.record.importance}")
        print(f"Reasons: {match.match_reasons}")
        print(f"Evidence gaps: {match.evidence_gaps}")


# ---------------------------------------------------------------------------
# 1. Basic remember()
# ---------------------------------------------------------------------------

def demonstrate_basic_memory() -> None:
    heading("1. Basic remember()")

    # Scope, categories, and importance are inferred by the LLM.
    record = memory.remember(
        content=(
            "The development team prefers small pull requests with unit tests."
        ),
        root_scope="/demo",
        source="developer-notes",
    )

    show_record(record)


# ---------------------------------------------------------------------------
# 2. Controlled memory
# ---------------------------------------------------------------------------

def demonstrate_controlled_memory() -> None:
    heading("2. Controlled memory")

    record = memory.remember(
        content="The TaskFlow backend uses Python 3.11 and FastAPI.",
        scope="/demo/project/technology",
        categories=["project", "python", "fastapi"],
        importance=0.9,
        metadata={
            "project": "TaskFlow",
            "verified": True,
        },
        source="project-documentation",
    )

    show_record(record)


# ---------------------------------------------------------------------------
# 3. Batch memory and deduplication
# ---------------------------------------------------------------------------

def demonstrate_batch_memory() -> None:
    heading("3. remember_many() and batch deduplication")

    records = memory.remember_many([
        "PostgreSQL is the production database.",
        "SQLite is used only for unit tests.",
        "The production database is PostgreSQL.",
    ])

    print(f"Records stored from batch: {len(records)}")

    for record in records:
        print(f"- {record.scope}: {record.content}")


# ---------------------------------------------------------------------------
# 4. Root scope
# ---------------------------------------------------------------------------

def demonstrate_root_scope() -> None:
    heading("4. Root scope")

    record = memory.remember(
        content="The login endpoint uses JWT authentication.",
        scope="/authentication",
        root_scope="/demo/taskflow",
        categories=["authentication", "jwt"],
        importance=0.85,
        source="architecture-document",
    )

    show_record(record)


# ---------------------------------------------------------------------------
# 5. Importance
# ---------------------------------------------------------------------------

def demonstrate_importance() -> None:
    heading("5. Importance")

    critical_record = memory.remember(
        content="Production must never use the test payment gateway.",
        scope="/demo/project/constraints",
        categories=["production", "payment", "constraint"],
        importance=1.0,
        source="security-policy",
    )

    general_record = memory.remember(
        content="The development team normally meets on Tuesday.",
        scope="/demo/project/general",
        categories=["team", "meeting"],
        importance=0.3,
        source="team-notes",
    )

    print("Critical record:")
    show_record(critical_record)

    print("\nGeneral record:")
    show_record(general_record)


# ---------------------------------------------------------------------------
# 6. Metadata and agent role
# ---------------------------------------------------------------------------

def demonstrate_metadata_and_agent_role() -> None:
    heading("6. Metadata and agent role")

    record = memory.remember(
        content=(
            "An unbounded cache caused the inventory API memory problem."
        ),
        scope="/demo/incidents/memory",
        categories=["incident", "memory", "cache"],
        importance=0.95,
        metadata={
            "incident_id": "INC-101",
            "server": "APP-01",
            "verified": True,
        },
        source="incident-operator",
        agent_role="System Monitor",
    )

    show_record(record)


# ---------------------------------------------------------------------------
# 7. Shallow recall
# ---------------------------------------------------------------------------

def demonstrate_shallow_recall() -> None:
    heading("7. Shallow recall")

    matches = memory.recall(
        query="Which technology does TaskFlow use?",
        scope="/demo",
        depth="shallow",
        limit=3,
    )

    show_matches(matches)


# ---------------------------------------------------------------------------
# 8. Deep recall
# ---------------------------------------------------------------------------

def demonstrate_deep_recall() -> None:
    heading("8. Deep recall")

    matches = memory.recall(
        query=(
            "What existing technical decisions could affect migrating "
            "TaskFlow to a different database?"
        ),
        scope="/demo",
        depth="deep",
        limit=5,
    )

    show_matches(matches)


# ---------------------------------------------------------------------------
# 9. Category filtering
# ---------------------------------------------------------------------------

def demonstrate_category_filter() -> None:
    heading("9. Category filtering")

    matches = memory.recall(
        query="What production rules must be followed?",
        scope="/demo",
        categories=["production"],
        depth="shallow",
        limit=5,
    )

    show_matches(matches)


# ---------------------------------------------------------------------------
# 10. Source filtering
# ---------------------------------------------------------------------------

def demonstrate_source_filter() -> None:
    heading("10. Source filtering")

    matches = memory.recall(
        query="What security rules must be followed?",
        source="security-policy",
        depth="shallow",
        limit=5,
    )

    show_matches(matches)


# ---------------------------------------------------------------------------
# 11. Private memory
# ---------------------------------------------------------------------------

def demonstrate_private_memory() -> None:
    heading("11. Private memory")

    private_record = memory.remember(
        content=(
            "An internal review found an insecure development configuration."
        ),
        scope="/demo/security/internal",
        categories=["security", "internal"],
        importance=0.9,
        source="security-review",
        private=True,
    )

    print("Private record:")
    show_record(private_record)

    print("\nNormal recall excludes private records:")

    public_matches = memory.recall(
        query="What insecure configuration was discovered?",
        scope="/demo/security",
        depth="shallow",
        include_private=False,
    )

    show_matches(public_matches)

    print("\nRecall explicitly including private records:")

    private_matches = memory.recall(
        query="What insecure configuration was discovered?",
        scope="/demo/security",
        depth="shallow",
        include_private=True,
    )

    show_matches(private_matches)


# ---------------------------------------------------------------------------
# 12. Scope view
# ---------------------------------------------------------------------------

def demonstrate_scope_view() -> None:
    heading("12. MemoryScope")

    server_memory = memory.scope("/demo/servers")

    record = server_memory.remember(
        content="APP-01 runs the inventory API on port 8080.",
        scope="/app-01/configuration",
        categories=["server", "configuration"],
        importance=0.8,
        source="server-documentation",
    )

    print("The scoped view automatically prefixes the path:")
    show_record(record)

    matches = server_memory.recall(
        query="Which port does the inventory API use?",
        depth="shallow",
    )

    show_matches(matches)


# ---------------------------------------------------------------------------
# 13. Read-only memory slice
# ---------------------------------------------------------------------------

def demonstrate_read_only_slice() -> None:
    heading("13. Read-only MemorySlice")

    memory.remember(
        content="Passwords must contain at least 12 characters.",
        scope="/demo/company/security",
        categories=["security", "password"],
        importance=1.0,
        source="company-policy",
    )

    memory.remember(
        content="Production deployments require manager approval.",
        scope="/demo/company/deployment-policies",
        categories=["deployment", "approval"],
        importance=1.0,
        source="company-policy",
    )

    policy_view = memory.slice(
        [
            "/demo/company/security",
            "/demo/company/deployment-policies",
        ],
        read_only=True,
    )

    print("Recall from multiple selected branches:")

    matches = policy_view.recall(
        query="What company rules must be followed?",
        depth="shallow",
        limit=5,
    )

    show_matches(matches)

    print("\nAttempting to write through a read-only slice:")

    result = policy_view.remember(
        content="Attempt to add an unauthorized policy."
    )

    print(f"remember() result: {result}")


# ---------------------------------------------------------------------------
# 14. Scope tree
# ---------------------------------------------------------------------------

def demonstrate_scope_tree() -> None:
    heading("14. Memory scope tree")

    print(memory.tree(max_depth=6))


# ---------------------------------------------------------------------------
# 15. List child scopes
# ---------------------------------------------------------------------------

def demonstrate_list_scopes() -> None:
    heading("15. List scopes")

    scopes = memory.list_scopes("/demo")

    if not scopes:
        print("No child scopes found.")
        return

    for scope in scopes:
        print(scope)


# ---------------------------------------------------------------------------
# 16. Scope information
# ---------------------------------------------------------------------------

def demonstrate_scope_info() -> None:
    heading("16. Scope information")

    scope_info = memory.info("/demo")

    print(scope_info)


# ---------------------------------------------------------------------------
# 17. MemoryRecord
# ---------------------------------------------------------------------------

def demonstrate_memory_record() -> None:
    heading("17. MemoryRecord")

    record = memory.remember(
        content="The API health-check endpoint is /health.",
        scope="/demo/project/api",
        categories=["api", "health-check"],
        importance=0.8,
        source="api-documentation",
    )

    show_record(record)


# ---------------------------------------------------------------------------
# 18. MemoryMatch
# ---------------------------------------------------------------------------

def demonstrate_memory_match() -> None:
    heading("18. MemoryMatch")

    matches = memory.recall(
        query="What is the API health-check endpoint?",
        scope="/demo/project",
        depth="shallow",
        limit=3,
    )

    show_matches(matches)


# ---------------------------------------------------------------------------
# 19. Crew with memory=True
# ---------------------------------------------------------------------------

def demonstrate_default_crew_memory() -> None:
    heading("19. Default crew memory with memory=True")

    examiner = Agent(
        role="Bug Examiner",
        goal="Identify the likely cause of a software problem.",
        backstory="You specialize in diagnosing Python API problems.",
        llm=openrouter_llm,
    )

    task = Task(
        description=(
            "Examine this bug: the login endpoint returns plain text "
            "instead of a structured JSON error."
        ),
        expected_output="A short diagnosis.",
        agent=examiner,
    )

    crew = Crew(
        agents=[examiner],
        tasks=[task],
        process=Process.sequential,
        memory=True,
        verbose=True,
    )

    result = crew.kickoff()

    print("\nCrew result:")
    print(result)


# ---------------------------------------------------------------------------
# 20. Crew with custom shared memory
# ---------------------------------------------------------------------------

def demonstrate_custom_crew_memory() -> None:
    heading("20. Custom crew-wide shared memory")

    custom_memory = Memory(
        llm=openrouter_llm,
        semantic_weight=0.4,
        recency_weight=0.4,
        importance_weight=0.2,
        recency_half_life_days=14,
    )

    custom_memory.remember(
        content=(
            "API errors must be JSON objects containing code and message."
        ),
        scope="/project/api-rules",
        categories=["api", "error-handling"],
        importance=1.0,
        source="api-standard",
    )

    bug_examiner = Agent(
        role="Bug Examiner",
        goal="Diagnose API bugs using existing project knowledge.",
        backstory="You specialize in Python API diagnosis.",
        llm=openrouter_llm,
        # Inherits crew memory.
    )

    fix_planner = Agent(
        role="Fix Planner",
        goal="Create fixes that follow existing project standards.",
        backstory="You specialize in safe software fixes.",
        llm=openrouter_llm,
        # Inherits crew memory.
    )

    examination_task = Task(
        description=(
            "Examine this bug: the login endpoint returns plain text "
            "when authentication fails. Use remembered API standards."
        ),
        expected_output="A short diagnosis.",
        agent=bug_examiner,
    )

    fix_task = Task(
        description=(
            "Create a fix for the diagnosed problem. Follow remembered "
            "project standards."
        ),
        expected_output="A short fix plan.",
        agent=fix_planner,
        context=[examination_task],
    )

    crew = Crew(
        agents=[bug_examiner, fix_planner],
        tasks=[examination_task, fix_task],
        process=Process.sequential,
        memory=custom_memory,
        verbose=True,
    )

    result = crew.kickoff()

    print("\nCrew result:")
    print(result)

    # Explicitly save important information only after verification.
    verified = input(
        "\nWas this fix verified by a developer? [y/N]: "
    ).strip().lower()

    if verified == "y":
        record = custom_memory.remember(
            content=f"Verified login error fix: {result}",
            scope="/project/verified-fixes",
            categories=["login", "verified-fix"],
            importance=0.95,
            source="developer",
            metadata={"verified": True},
        )

        print("\nVerified result saved:")
        show_record(record)
    else:
        print("The unverified result was not explicitly saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_memory_demonstrations(include_deep: bool) -> None:
    demonstrate_basic_memory()
    demonstrate_controlled_memory()
    demonstrate_batch_memory()
    demonstrate_root_scope()
    demonstrate_importance()
    demonstrate_metadata_and_agent_role()
    demonstrate_shallow_recall()

    if include_deep:
        demonstrate_deep_recall()
    else:
        heading("8. Deep recall")
        print("Skipped. Run with --deep to execute deep recall.")

    demonstrate_category_filter()
    demonstrate_source_filter()
    demonstrate_private_memory()
    demonstrate_scope_view()
    demonstrate_read_only_slice()
    demonstrate_scope_tree()
    demonstrate_list_scopes()
    demonstrate_scope_info()
    demonstrate_memory_record()
    demonstrate_memory_match()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate the unified CrewAI Memory API."
    )

    parser.add_argument(
        "--deep",
        action="store_true",
        help="Run the more expensive LLM-assisted deep recall.",
    )

    parser.add_argument(
        "--crew",
        action="store_true",
        help="Run the additional crew-level memory demonstrations.",
    )

    args = parser.parse_args()

    run_memory_demonstrations(include_deep=args.deep)

    if args.crew:
        demonstrate_default_crew_memory()
        demonstrate_custom_crew_memory()
    else:
        heading("Crew memory demonstrations")
        print("Skipped. Run with --crew to execute the crew examples.")


if __name__ == "__main__":
    main()