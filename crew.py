import os
import streamlit as st
import requests
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Optional

# 1. Environment & API Setup
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# 2. STRICT SCHEMA FOR GROQ
# Groq requires 'additionalProperties: false' on EVERY object in the schema.
class ApiCallerInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    url: str = Field(..., description="The full URL of the API endpoint")
    method: str = Field(..., description="HTTP method: GET, POST, PUT, or DELETE")
    headers: Dict[str, Any] = Field(default_factory=dict, description="Headers as a flat key-value dictionary")
    json_body: Dict[str, Any] = Field(default_factory=dict, description="JSON body as a flat key-value dictionary")

class ApiCallerTool(BaseTool):
    name: str = "api_caller_tool"
    description: str = "Use this to execute real REST API calls. Provide url, method, headers, and json_body."
    args_schema: type[BaseModel] = ApiCallerInput

    def _run(self, url: str, method: str, headers: Optional[Dict] = None, json_body: Optional[Dict] = None) -> str:
        try:
            h = headers if isinstance(headers, dict) else {}
            b = json_body if isinstance(json_body, dict) else {}
            
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=h,
                json=b if method.upper() in ["POST", "PUT", "PATCH"] else None,
                timeout=15
            )
            return f"Status: {response.status_code}\nResponse: {response.text[:2000]}"
        except Exception as e:
            return f"Error: {str(e)}"

# Instantiate the tool
api_caller_tool = ApiCallerTool()

@CrewBase
class ApiTestingCrew():
    agents_config = 'agents.yaml'
    tasks_config = 'tasks.yaml'

    def __init__(self):
        self.groq_llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=groq_api_key
        )

    @agent
    def api_requirement_collector(self) -> Agent:
        return Agent(
            config=self.agents_config['api_requirement_collector'],
            llm=self.groq_llm,
            verbose=True
        )

    @agent
    def api_executor(self) -> Agent:
        return Agent(
            config=self.agents_config['api_executor'],
            tools=[api_caller_tool],
            llm=self.groq_llm,
            verbose=True,
            allow_delegation=False # Prevents Groq from getting confused by nested calls
        )

    @agent
    def test_result_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['test_result_analyst'],
            llm=self.groq_llm,
            verbose=True
        )

    # TASKS - Matching the keys in your tasks.yaml
    @task
    def collection_task(self) -> Task:
        return Task(
            config=self.tasks_config['input_collection_task'],
            agent=self.api_requirement_collector()
        )

    @task
    def execution_task(self) -> Task:
        return Task(
            config=self.tasks_config['api_execution_task'],
            agent=self.api_executor()
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            agent=self.test_result_analyst(),
            output_file='api_report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
