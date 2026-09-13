from strands import Agent
import boto3
import json

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)


MEMORY_ID = "agentsforhumanscore_AgentsForHumansMemory-KI3pyT4cTg"
MEMORY_REGION = "us-west-2"
PREFERENCE_STRATEGY_ID = "AgentsForHumansCustomPreference-o2t5Q16GkM"
SEMANTIC_STRATEGY_ID = "AgentsForHumansMemory_Semantic-WI5kbHCmOj"


def load_baseline_profile() -> dict:
    """Load the user's baseline media profile from the local JSON file."""
    with open("data/sample_profile.json", "r", encoding="utf-8") as file:
        return json.load(file)


BASELINE_PROFILE = load_baseline_profile()


def load_current_preferences(actor_id: str) -> list[dict]:
    """
    Load all USER_PREFERENCE memory records for the current actor.

    This uses ListMemoryRecords instead of semantic retrieval so important
    preferences are not omitted just because they rank poorly for a generic query.
    """

    client = boto3.client(
        "bedrock-agentcore",
        region_name=MEMORY_REGION,
    )

    namespace = f"/users/{actor_id}/preferences"
    records = []
    next_token = None

    while True:
        # First page: no nextToken is needed.
        if next_token is None:
            response = client.list_memory_records(
                memoryId=MEMORY_ID,
                namespace=namespace,
                memoryStrategyId=PREFERENCE_STRATEGY_ID,
                maxResults=100,
            )

        # Later pages: send back the nextToken AgentCore returned.
        else:
            response = client.list_memory_records(
                memoryId=MEMORY_ID,
                namespace=namespace,
                memoryStrategyId=PREFERENCE_STRATEGY_ID,
                maxResults=100,
                nextToken=next_token,
            )

        records.extend(response.get("memoryRecordSummaries", []))

        next_token = response.get("nextToken")
        if not next_token:
            break

    return records


SYSTEM_PROMPT = f"""
You are a personal media discovery agent.

Your job is to help a user discover media across categories such as music,
podcasts, audiobooks, videos, and books. Recommendations should be personalized
using the user's initial media profile and relevant memories about the user.

The sample media profile is baseline seed data and may become outdated over time.
Current memory should be treated as the source of truth when it conflicts with
the sample profile.

When responding:

1. Use the baseline media profile when the user's preferences are relevant.
2. Use remembered information when it is relevant to the request.
3. Base personalization on specific evidence.
4. Do not claim the user likes something unless the profile or memory supports it.
5. Prefer recommendations that connect multiple signals rather than simply
   repeating media the user already knows.
6. Explain briefly why each recommendation fits.
7. When possible, introduce some novelty instead of recommending only the
   user's existing favorite creators.
8. If there is not enough information to make a confident recommendation,
   say what additional preference would help.

MEMORY PRIORITY RULES:

- The sample media profile is baseline seed data, not a current preference snapshot.
- User preference memories represent the user's current tastes and recommendation constraints.
- Semantic/fact memories provide background context and may describe interests the user held in the past.
- If a current preference memory conflicts with the sample profile, follow the preference memory.
- If a current preference memory conflicts with a semantic/fact memory, follow the preference memory.
- Explicit negative preferences such as "does not want", "no longer likes", or "do not recommend" override older positive information about the same topic.
- The presence of an interest in the sample profile must NEVER be interpreted as the user re-adding, reaffirming, or reversing a newer preference stored in memory.
- Never recommend content that conflicts with an explicit current negative preference.
- Explicit preference statements made in the current conversation take effect immediately.
- Do not wait for long-term memory extraction before following a preference the user just stated.
- If the current conversation conflicts with USER_PREFERENCE memory or the baseline profile, follow the user's newest explicit statement in the current conversation.

For a ranked recommendation queue, use this structure:

[
    {{
        "placeInQueue": 1,
        "publisherName": "...",
        "mediaTitle": "...",
        "mediaType": "song | podcast | audiobook | video | book",
        "explanationForRecommendation": "...",
        "categories": ["...", "..."]
    }}
]

IMPORTANT:

The sample profile is historical baseline data. It may contain preferences that
the user has since changed.

Do not treat an item appearing in the sample profile as evidence that the user
currently likes it if memory contains a newer contradictory preference.

The sample profile also does not prove that a specific recommendation is current,
newly released, or available on a platform. Do not invent release dates, chart
positions, availability, or other current facts.

BASELINE MEDIA PROFILE:

{json.dumps(BASELINE_PROFILE, indent=2)}

END BASELINE MEDIA PROFILE.

Current memory should be treated as the source of truth when it conflicts with
the baseline profile.
"""


app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload, context):
    prompt = payload.get("prompt", "")
    session_id = context.session_id
    actor_id = payload.get("actor_id", "")
    if not actor_id:
        raise ValueError("actor_id is required")

    current_preferences = load_current_preferences(actor_id)

    # Sort preferences from oldest to newest.
    sorted_preferences = sorted(
        current_preferences,
        key=lambda record: record.get("createdAt", ""),
    )

    current_preferences_text = "\n\n".join(
        (
            f"Created at: {record.get('createdAt', 'unknown')}\n"
            f"{record.get('content', {}).get('text', '')}"
        )
        for record in sorted_preferences
    )

    runtime_system_prompt = f"""
    {SYSTEM_PROMPT}

    CURRENT USER PREFERENCE MEMORY:

    {current_preferences_text}

    END CURRENT USER PREFERENCE MEMORY.

    These USER_PREFERENCE records represent the user's current preferences and
    recommendation constraints.

    The records are listed from oldest to newest.

    If two USER_PREFERENCE records conflict about the same preference:
    - follow the newer record
    - treat the older conflicting record as superseded
    - do not enforce an older negative preference after the user has explicitly reversed it

    USER_PREFERENCE records override conflicting baseline media profile information.
    """

    # Configure semantic long-term memory retrieval.
    retrieval_config = {
        f"/users/{actor_id}/facts": RetrievalConfig(
            # Find up to 2 strongly relevant semantic facts.
            top_k=2,
            relevance_score=0.7,
            strategy_id=SEMANTIC_STRATEGY_ID,
            ),
    }

    # Configure AgentCore memory for this request/session.
    agentcore_memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config=retrieval_config,
    )

    # Connect Strands session handling to AgentCore Memory.
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=agentcore_memory_config,
        region_name=MEMORY_REGION,
    )

    # Create the Strands agent.
    agent = Agent(
        system_prompt=runtime_system_prompt,
        session_manager=session_manager,
    )

    response = agent(prompt)

    return {
        "result": str(response),
        "session_id": session_id,
        "actor_id": actor_id,
    }


if __name__ == "__main__":
    app.run()