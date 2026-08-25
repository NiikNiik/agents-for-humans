from strands import Agent, tool
import json
from bedrock_agentcore.runtime import BedrockAgentCoreApp

@tool
def get_media_profile() -> dict:
    """Load the user's sample media profile.

    Use this tool whenever you need evidence about the user's media preferences,
    including favorite artists, liked songs, followed podcasts, genres, topics,
    or other interests contained in the sample profile.

    Do not invent preferences that are not supported by the profile.

    Returns:
        dict: The contents of data/sample_profile.json.
    """
    with open("data/sample_profile.json", "r", encoding="utf-8") as file:
        return json.load(file)


SYSTEM_PROMPT = """
You are a personal media discovery agent.

Your job is to help a user discover media across categories such as music,
podcasts, audiobooks, videos, and books. Recommendations should be personalized
using evidence from the user's media profile.

When responding:

1. Use get_media_profile when the user's preferences are relevant.
2. Base personalization on specific evidence returned by the tool.
3. Do not claim the user likes something unless the profile supports it.
4. Prefer recommendations that connect multiple signals rather than simply
   repeating media the user already knows.
5. Explain briefly why each recommendation fits.
6. When possible, introduce some novelty instead of recommending only the
   user's existing favorite creators.
7. If there is not enough information to make a confident recommendation,
   say what additional preference would help.

For a ranked recommendation queue, use this structure:

[
    {
        "placeInQueue": 1,
        "publisherName": "...",
        "mediaTitle": "...",
        "mediaType": "song | podcast | audiobook | video | book",
        "explanationForRecommendation": "...",
        "categories": ["...", "..."]
    }
]

IMPORTANT:
The sample profile tells you what the user likes, but it does not prove that a
specific recommendation is current, newly released, or available on a platform.
Do not invent release dates, chart positions, availability, or other current
facts.

Later versions of this agent will add:
- AgentCore Memory to retain feedback/preferences across runs.
- HTTP/API tools for current media discovery.
- handoff_to_user for confirmation before actions such as adding items to a queue.
"""


app = BedrockAgentCoreApp()

agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=[get_media_profile],
)


@app.entrypoint
def invoke(payload):
    """Process user input and return a response."""
    user_message = payload.get("prompt", "Hello")
    result = agent(user_message)
    return {"result": result.message}        

if __name__ == "__main__":
    app.run()
