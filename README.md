# Drift

Drift is a personal media discovery agent that learns a user's preferences across conversations and uses that evolving context to make recommendations across podcasts, books, music, videos, and audiobooks.

Built for the **Agents for Humans** hackathon using the **Strands Agents SDK**, **Amazon Bedrock AgentCore Runtime**, and **Amazon Bedrock AgentCore Memory**.

## Try It

Public demo: https://niikniik-agents-for-humans-streamlit-app-agentcore-runti-phbja2.streamlit.app/

The demo starts with a sample media profile. Preferences expressed in conversation can change over time and override that baseline profile.

## Why Drift

Most media platforms learn only one part of a user's taste. Music services know listening history, podcast apps know followed shows, and reading platforms know books, but those signals usually remain isolated.

Drift explores a different model: one personal agent that can combine signals across media types and continue learning as the user's interests change.

The prototype focuses on three kinds of context:

1. **Baseline media profile** — historical seed data from `data/sample_profile.json`.
2. **Current preferences** — explicit preferences and recommendation constraints stored with AgentCore `USER_PREFERENCE` memory.
3. **Background facts** — relevant long-term context stored with AgentCore `SEMANTIC` memory.

The central recommendation rule is:

> Baseline data tells Drift where the user started. Current preferences tell Drift what the user wants now.

Newer explicit preferences override conflicting baseline information.

## Architecture

![Drift architecture](docs/drift-architecture.png)

### Request flow

The Streamlit application creates a stable per-visitor `actor_id` in browser local storage and a separate `session_id` for each conversation. It sends the prompt, actor ID, and session ID to the deployed AgentCore Runtime.

Inside the runtime, Drift loads the baseline profile, retrieves current `USER_PREFERENCE` records, configures semantic memory retrieval, and creates a Strands agent using an AgentCore-backed session manager. The agent then generates a personalized recommendation response and returns it to the Streamlit UI.

## Memory Design

Persistent memory was the most important architectural problem in this project.

The first implementation exposed the sample media profile to the agent through a normal tool result. Because that tool output became part of conversational history, imported profile information could later influence preference memory as if it were current user evidence.

The design was changed so that baseline profile information is loaded directly as baseline context while actual user preference changes continue to flow into AgentCore Memory.

For recommendation requests, Drift:

- loads the baseline media profile;
- loads current `USER_PREFERENCE` records comprehensively;
- retrieves relevant `SEMANTIC` facts;
- follows the newest explicit preference when information conflicts.

This lets Drift preserve useful historical context without allowing old profile data to permanently define the user.

## Project Structure

- `agent.py` — Drift agent, AgentCore Runtime entry point, memory configuration, and recommendation logic.
- `streamlit_app.py` — public Streamlit chat interface and AgentCore Runtime invocation.
- `data/sample_profile.json` — baseline sample media profile used by the prototype.
- `agentcore/` — AgentCore deployment/configuration files.
- `docs/` — project documentation and architecture materials.
- `requirements.txt` — Python dependencies.

## Technology

- Python 3.12
- Strands Agents SDK
- Amazon Bedrock AgentCore Runtime
- Amazon Bedrock AgentCore Memory
- Amazon Bedrock
- Boto3
- Streamlit

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/NiikNiik/agents-for-humans.git
cd agents-for-humans
git checkout agentcore-runtime
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure AWS credentials

You need AWS credentials with access to the Bedrock and AgentCore resources used by your deployment. Configure the AWS CLI or another Boto3-supported credential source, and verify the intended AWS region before invoking or deploying resources.

The hackathon deployment uses `us-west-2`.

### 5. Run the agent locally

```bash
python agent.py
```

`agent.py` exposes the Bedrock AgentCore application entry point used by the deployed runtime.

### 6. Run the Streamlit UI

The Streamlit frontend invokes a deployed AgentCore Runtime. If you are using your own deployment, update the runtime ARN in `streamlit_app.py` to point to your runtime.

```bash
streamlit run streamlit_app.py
```

## Testing the Memory Behavior

A useful way to verify cross-session personalization is:

1. Start one conversation and state a new preference, for example that you are currently interested in a particular genre or topic.
2. Start a new conversation.
3. Ask Drift for recommendations based on what it knows about you.
4. Verify that the newer preference affects the recommendations despite the new session.

Preference-reversal tests can also be used to confirm that newer explicit preferences supersede conflicting older information.

## Current Scope

The current prototype uses a sample profile instead of connecting directly to external media platforms. It also relies primarily on the model's existing media knowledge rather than live discovery APIs.

Future work could include real integrations for music, podcasts, books, audiobooks, and video services; live discovery/search; and user controls for reviewing or correcting the unified profile Drift builds.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
