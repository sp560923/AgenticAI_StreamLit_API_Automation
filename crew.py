import os
import streamlit as st
import requests
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

# -------------------------------------------------------------------
# 1. Environment & Safety Settings (IMPORTANT FOR STREAMLIT + GROQ)
# -------------------------------------------------------------------
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# Disable LiteLLM proxy / SSO / logging paths
os.environ["LITELLM_MODE"] = "STANDARD"
os.environ["LITELLM_DISABLE_LOGGING"] = "true"
os.environ["LITELLM_PROXY_DISABLED"] = "true"

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# -------------------------------------------------------------------
# 2. GROQ-STRICT TOOL SCHEMA (NO additionalProperties VIOLATION)
# -------------------------------------------------------------------

class KeyValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(..., description="Header or JSON key")
    value: str = Field(..., description="Header or JSON value")

class ApiCallerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Full API endpoint URL")
    method: str = Field(..., description="HTTP method: GET, POST, PUT, DELETE")
    headers: List[KeyValue] = Field(default_factory=list)
    json_body: List[KeyValue] = Field(default_factory=list)

# -------------------------------------------------------------------
# 3. API CALLER TOOL
# -------------------------------------------------------------------

class ApiCallerTool(BaseTool):
    name: str = "api_caller_tool"
    description: str = "Executes real REST API calls using provided URL, method, headers, and JSON body."
    args_schema: type[BaseModel] = ApiCallerInput

    def _run(
        self,
        url: str,
        method: str,
        headers: Optional[list] = None,
        json_body: Optional[list] = None
    ) -> str:
        try:
            headers_dict = {h["key"]: h["value"] for h in (headers or [])}
            body_dict = {b["key"]: b["value"] for b in (json_body or [])}

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
# 4. CREW DEFINITION
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

