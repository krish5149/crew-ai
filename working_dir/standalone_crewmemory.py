import os
from crewai import Memory, LLM
from dotenv import load_dotenv

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

memory_management = Memory(
    llm=openrouter_llm,
    embedder={
        "provider": "sentence-transformer",
        "config": {
            "model": "all-MiniLM-L6-v2"
            }
        }
    )

def add_learning_memory():

    content = input("What should be remembered ?")

    if not content:
        print("Memory content cannot be empty.")
        return

    record = memory_management.remember(
        content = content,
        root_scope = "/learning",
        source = "user"
    )

    if record is None:
        print("The memory was not added, possibly because it was a duplicate.")
        return

    print("\nMemory saved")
    print(f"ID: {record.id}")
    print(f"Scope: {record.scope}")
    print(f"Categories: {record.categories}")
    print(f"Importance: {record.importance}")
    print(f"Content: {record.content}")

def add_controlled_memory():

    topic = input("\nTopic name: ").strip()
    note = input("Learning note: ").strip()

    if not topic or not note:
        print("Topic and learning note are required.")
        return

    normalized_topic = topic.lower().replace(" ", "-")

    record = memory_management.remember(
        content = note,
        scope = "/learning/normalized_topic/{normalized_topic}",
        categories = ["learning","topic",normalized_topic],
        importance = 0.8,
        metadata = {"topic":topic},
        source = "user deatiled"
    )

    if record is None:
        print("The memory was not added, possibly because it was a duplicate.")
        return

    print(f"\nSaved under: {record.scope}")



def recall_memories() -> None:
    query = input("\nWhat do you want to recall? ").strip()

    if not query:
        print("A recall question is required.")
        return

    depth_choice = input(
        "Recall depth [shallow/deep, default=shallow]: "
    ).strip().lower()

    depth = "deep" if depth_choice == "deep" else "shallow"

    matches = memory_management.recall(
        query=query,
        scope="/learning",
        limit=5,
        depth=depth,
    )

    if not matches:
        print("\nNo relevant memories found.")
        return

    print(f"\nFound {len(matches)} relevant memories:")

    for number, match in enumerate(matches, start=1):
        record = match.record

        print(f"\n{number}. Score: {match.score:.3f}")
        print(f"   Content: {record.content}")
        print(f"   Scope: {record.scope}")
        print(f"   Categories: {record.categories}")
        print(f"   Importance: {record.importance}")
        print(f"   Match reasons: {match.match_reasons}")

        if match.evidence_gaps:
            print(f"   Evidence gaps: {match.evidence_gaps}")


def show_memory_tree() -> None:
    print("\nMemory scope tree:")
    print(memory_management.tree(max_depth=5))

def show_menu() -> None:
    print(
        """
Standalone Learning Memory

1. Add an LLM-organized memory
2. Add a controlled topic memory
3. Add sample memories
4. Recall relevant memories
5. Show memory scope tree
6. Show scope information
0. Exit
"""
    )


def main() -> None:
    while True:
        show_menu()
        choice = input("Select an option: ").strip()

        if choice == "1":
            add_learning_memory()
        elif choice == "2":
            add_controlled_memory()
        elif choice == "3":
            recall_memories()
        elif choice == "5":
            show_memory_tree()
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print("Invalid option. Select a number from 0 to 6.")


if __name__ == "__main__":
    main()