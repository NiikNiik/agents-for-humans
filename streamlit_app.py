import json
import uuid

import boto3
import streamlit as st
from streamlit_local_storage import LocalStorage


RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-west-2:471112779923:"
    "runtime/agentsforhumanscore_RecommendationAgent-UUT3f87YaO"
)

AWS_REGION = "us-west-2"
ACTOR_STORAGE_KEY = "drift_actor_id"


st.set_page_config(
    page_title="Drift",
    page_icon="🧭",
)


local_storage = LocalStorage()


def new_session_id() -> str:
    return f"drift-session-{uuid.uuid4()}"


def get_actor_id() -> str:
    if "actor_id" in st.session_state:
        return st.session_state.actor_id

    stored_actor_id = local_storage.getItem(ACTOR_STORAGE_KEY)

    if stored_actor_id:
        actor_id = stored_actor_id
    else:
        actor_id = f"drift-visitor-{uuid.uuid4()}"
        local_storage.setItem(ACTOR_STORAGE_KEY, actor_id)

    st.session_state.actor_id = actor_id
    return actor_id


def invoke_agent(prompt: str, actor_id: str, session_id: str) -> str:
    client = boto3.client(
        "bedrock-agentcore",
        region_name=AWS_REGION,
    )

    payload = {
        "prompt": prompt,
        "actor_id": actor_id,
    }

    response = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )

    body = json.loads(response["response"].read())

    return body["result"]


actor_id = get_actor_id()

if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("Drift")

st.write(
    "A personal media discovery agent that learns your preferences across conversations."
)

st.caption(
    "Demo note: Drift starts with a sample media profile. "
    "Your preferences can change over time and override that starting profile."
)


if st.button("New Conversation"):
    st.session_state.session_id = new_session_id()
    st.session_state.messages = []
    st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input("Ask Drift for a recommendation...")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = invoke_agent(
                    prompt=prompt,
                    actor_id=actor_id,
                    session_id=st.session_state.session_id,
                )
            except Exception as exc:
                st.error(f"Agent request failed: {exc}")
                st.stop()

        st.write(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result,
        }
    )