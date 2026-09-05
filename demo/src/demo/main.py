import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from demo.crews.content_crew.ResearchCrew import ResearchCrew

load_dotenv()


def _inputs() -> dict[str, str]:
    """Inputs interpolated into the {placeholders} in the YAML config."""
    return {
        "topic": "AI agent frameworks",
        "current_year": str(datetime.now(timezone.utc).year),
    }


def run() -> None:
    """Run the crew once."""
    result = ResearchCrew().crew().kickoff(inputs=_inputs())
    print(result.raw)        #type: ignore


def train() -> None:
    """Train the crew for a given number of iterations.

    Usage: train <n_iterations> <filename>
    """
    try:
        ResearchCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=_inputs(),
        )
    except IndexError:
        raise SystemExit("usage: train <n_iterations> <filename>")


def replay() -> None:
    """Replay execution from a specific task.

    Usage: replay <task_id>
    """
    try:
        ResearchCrew().crew().replay(task_id=sys.argv[1])
    except IndexError:
        raise SystemExit("usage: replay <task_id>")


def test() -> None:
    """Evaluate the crew over N iterations with an eval LLM.

    Usage: test <n_iterations> <eval_llm>
    """
    try:
        ResearchCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=_inputs(),
        )
    except IndexError:
        raise SystemExit("usage: test <n_iterations> <eval_llm>")


if __name__ == "__main__":
    run()
