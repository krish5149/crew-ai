import os
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

llm = LLM(
    model = "openai/gpt-4o-mini",
    api_key = os.environ["OPENROUTER_API_KEY"],
    base_url = "https://openrouter.ai/api/v1",
    temperature = 0.7
)

search_tool = SerperDevTool()

reacher_agent = Agent(
    role = "Senior Research Analyst specializing in {topic}",
    goal = "Uncover, verify, and distill the most relevant, accurate, and recent "
           "information about {topic} into a structured set of findings that a writer "
           "or decision-maker can act on without doing further research. "
           "Every claim must be traceable to a source.",
    backstory = (
        "You are a veteran research analyst with over a decade of experience at a "
        "top-tier technology think tank, where you built a reputation for separating "
        "signal from hype. You are relentlessly curious and deeply skeptical: you never "
        "accept a claim at face value, you cross-check every important fact against at "
        "least two independent sources, and you explicitly flag anything that is "
        "speculative, contested, or outdated. You have a gift for taking a messy, "
        "contradictory pile of sources and turning it into a clear, well-organized brief — "
        "grouping findings by theme, leading with what matters most, and noting exactly "
        "where the evidence is thin. You never invent facts or citations; if you cannot "
        "find something, you say so plainly."
    ),
    tools = [search_tool],
    llm = llm,
    verbose = True,
    allow_delegation = False
)


writer_agent = Agent(
    role = "Senior Content Strategist and Technical Writer covering {topic}",
    goal = (
        "Write the final article on {topic} using ONLY the research brief provided in "
        "your context. Your job is composition, not discovery: restructure, sharpen, and "
        "explain what the researcher found. Do not add a single fact, figure, quote, date, "
        "or source that is not already in the brief. If the brief is missing something the "
        "article needs, write around it or state plainly that it is unknown."
    ),
    backstory = (
        "You spent years as a features editor at a respected technology publication, where "
        "your desk received finished analyst briefs and your job was to turn them into pieces "
        "people actually finished reading. You never did your own reporting — you had no "
        "sources of your own and you were fine with that, because your craft was structure "
        "and restraint: open with the one idea that matters, build each section on a single "
        "clear point, cut every sentence that is decoration rather than information. You write "
        "in plain, confident prose: short sentences, concrete nouns, active voice, no corporate "
        "filler, no hype, no 'in today's fast-paced world' openings.\n\n"
        "You treat the brief in front of you as the ONLY source of truth. You have no research "
        "tools and you do not fill gaps from memory — inventing a statistic or a citation was "
        "a firing offence at your old desk, and you have never done it. You carry every source "
        "from the brief through into the final text. When the brief marks something as "
        "uncertain or contested, you say so in the article instead of smoothing it over. You "
        "would rather ship a shorter piece that is entirely true than a longer one that is padded."
    ),
    llm = llm,
    verbose = True,
    allow_delegation = False
)

research_task = Task(
    description=(
        "Research {topic}. Find the most important developments, key players, "
        "concrete numbers, and open questions. Verify anything significant against "
        "at least two independent sources."
    ),
    expected_output=(
        "A bulleted brief of 8-12 findings grouped by theme. Each finding is one "
        "or two sentences followed by its source URL. Mark any contested or "
        "unverified item with [UNCERTAIN]."
    ),
    agent=reacher_agent,
)

write_task = Task(
    description=(
        "Write an article on {topic} for a smart non-expert audience, using only the "
        "research brief in your context.\n"
        "Rules:\n"
        "- Every factual claim must appear in the brief. No outside knowledge.\n"
        "- Carry the brief's source links through as inline markdown links.\n"
        "- Anything the brief marked [UNCERTAIN] must be hedged in the text.\n"
        "- If the brief does not cover something, omit it. Do not fill the gap."
    ),
    expected_output=(
        "A markdown article: headline, two-sentence lede, 4-6 sections, and a short "
        "'What to watch' close. Inline markdown citations. No facts beyond the brief."
    ),
    agent=writer_agent,
    context=[research_task],
)

crew = Crew(
    agents = [reacher_agent, writer_agent],
    tasks = [research_task, write_task],
    process = Process.sequential,
    verbose = True
)

result = crew.kickoff(inputs={"topic":"AT agents in 2026"})
print(result)