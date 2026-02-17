import os
import json
import streamlit as st
import requests
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any

# -------------------------------------------------------------------
# 1. Environment & Safety Settings (IMPORTANT FOR STREAMLIT + GROQ)
# -------------------------------------------------------------------
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

os.environ["LITELLM_MODE"] = "STANDARD"
os.environ["LITELLM_DISABLE_LOGGING"] = "true"
os.environ["LITELLM_PROXY_DISABLED"] = "true"

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# -------------------------------------------------------------------
# 2. GROQ-COMPATIBLE TOOL SCHEMA (✅ FIXED FOR NESTED JSON)
# -------------------------------------------------------------------

class KeyValue(BaseModel):
    """
    IMPORTANT:
    - value MUST allow Any
    - Groq validates tool args BEFORE _run()
    - Nested JSON bodies REQUIRE this
    """
    model_config = ConfigDict(extra="forbid")
    key: str = Field(..., description="Header or JSON key")
    value: Any = Field(
        ...,
        description="Header or JSON value (string, number, object, or list)"
    )

class ApiCallerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(..., description="Full API endpoint URL")
    method: str = Field(..., description="HTTP method: GET, POST, PUT, DELETE")
    headers: List[KeyValue] = Field(default_factory=list)
    json_body: List[KeyValue] = Field(default_factory=list)

# -------------------------------------------------------------------
# 3. 🔒 DEFENSIVE NORMALIZATION (CRITICAL – DO NOT REMOVE)
# -------------------------------------------------------------------

def enforce_string_kv(kv_list: Optional[list]) -> list:
    """
    Last-mile schema enforcement.
    Converts dict / list → JSON string.
    Ensures requests lib always receives valid payloads.
    """
    safe_list = []

    for item in kv_list or []:
        value = item.get("value")

        if isinstance(value, (dict, list)):
            safe_value = json.dumps(value)
        else:
            safe_value = str(value)

        safe_list.append({
            "key": str(item.get("key")),
            "value": safe_value
        })

    return safe_list

# -------------------------------------------------------------------
# 4. API CALLER TOOL (SINGLE + BULK SAFE)
# -------------------------------------------------------------------

class ApiCallerTool(BaseTool):
    name: str = "api_caller_tool"
    description: str = (
        "Executes real REST API calls using provided URL, method, "
        "headers, and JSON body."
    )
    args_schema: type[BaseModel] = ApiCallerInput

    def _run(
        self,
        url: str,
        method: str,
        headers: Optional[list] = None,
        json_body: Optional[list] = None
    ) -> str:
        try:
            # 🔐 Normalize tool inputs
            safe_headers = enforce_string_kv(headers)
            safe_body = enforce_string_kv(json_body)

            headers_dict = {h["key"]: h["value"] for h in safe_headers}
            body_dict = {b["key"]: b["value"] for b in safe_body}

            response = requests.request(
                method=method.upper(),
                url=url,
                headers=headers_dict,
                json=body_dict if method.upper() in ["POST", "PUT", "PATCH"] else None,
                timeout=15
            )

            return (
                f"Status Code: {response.status_code}\n"
                f"Response Headers: {dict(response.headers)}\n"
                f"Response Body:\n{response.text[:2000]}"
            )

        except Exception as e:
            return f"API Execution Error: {str(e)}"

# Instantiate tool
api_caller_tool = ApiCallerTool()

# -------------------------------------------------------------------
# 5. CREW DEFINITION
# -------------------------------------------------------------------

@CrewBase
class ApiTestingCrew():
    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"

    def __init__(self):
        self.groq_llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=groq_api_key
        )

    # ---------------- Agents ----------------

    @agent
    def api_requirement_collector(self) -> Agent:
        return Agent(
            config=self.agents_config["api_requirement_collector"],
            llm=self.groq_llm,
            verbose=True
        )

    @agent
    def api_executor(self) -> Agent:
        return Agent(
            config=self.agents_config["api_executor"],
            tools=[api_caller_tool],
            llm=self.groq_llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def test_result_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["test_result_analyst"],
            llm=self.groq_llm,
            verbose=True
        )

    # ---------------- Tasks ----------------

    @task
    def collection_task(self) -> Task:
        return Task(
            config=self.tasks_config["input_collection_task"],
            agent=self.api_requirement_collector()
        )

    @task
    def execution_task(self) -> Task:
        return Task(
            config=self.tasks_config["api_execution_task"],
            agent=self.api_executor()
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],
            agent=self.test_result_analyst(),
            output_file="api_report.md"
        )

    # ---------------- Crew ----------------

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
