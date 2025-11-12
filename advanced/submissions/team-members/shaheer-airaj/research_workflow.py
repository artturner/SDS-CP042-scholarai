import os
from agents import Agent, Runner, WebSearchTool, InputGuardrail, GuardrailFunctionOutput
from agents.exceptions import InputGuardrailTripwireTriggered
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ResearchValidation(BaseModel):
    is_valid: bool
    reason: str | None = None

class ResearchResults(BaseModel):
    findings: str

class Subtopics(BaseModel):
    subtopics: list[str]

input_guardrail_agent = Agent(
    name="InputGuardrail",
    instructions="You are an input validation agent. Ensure that the user input is asking to research a topic.",
    output_type=ResearchValidation
)

async def input_guardrail(ctx, agent, input_data):
    result = await Runner.run(input_guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ResearchValidation)
    return GuardrailFunctionOutput(
        output_info = final_output,
        tripwire_triggered = not final_output.is_valid
    )


topic_splitter_agent = Agent(
    name="TopicSplitterAgent",
    instructions="You are a topic splitting agent. Break down the user query to research a topic into 3 subtopics to be researched via the web.",
    output_type=Subtopics,
    input_guardrails=[InputGuardrail(guardrail_function=input_guardrail)]
)

research_agent = Agent(
    name="ResearchAgent",
    instructions="You are a research agent. Use web search to gather information on the given subtopics.",
    tools=[WebSearchTool()],
    output_type=ResearchResults
)

async def research_workflow(user_query: str) -> list[str]:

    # Step 1: Get subtopics
    try:
        subtopics = await Runner.run(topic_splitter_agent, user_query)
    except InputGuardrailTripwireTriggered as e:
        raise ValueError("Input validation failed: " + str(e))

    subtopic_list = subtopics.final_output_as(Subtopics).subtopics
    print(f"Identified subtopics: {subtopic_list}")

    # Step 2: Create research tasks for each subtopic
    research_tasks = [
        Runner.run(research_agent, subtopic) for subtopic in subtopic_list
    ]

    # Step 3: Await all research tasks to complete
    research_results = await asyncio.gather(*research_tasks)

    # step 3: Extract findings
    findings = [result.final_output_as(ResearchResults) for result in research_results]

    return findings


async def main():
    user_query = "Impact of rising house prices on millennial home ownership"
    research = await research_workflow(user_query)

    for idx, result in enumerate(research):
    print(f"\nSubtopic {idx + 1}:")
    print(result.findings)

if __name__ == "__main__":
    asyncio.run(main())