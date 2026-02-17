import os
import streamlit as st  # Fixes the NameError
import requests
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

# Disable telemetry to prevent the "signal" error in Streamlit
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
# Use the secret from Streamlit Dashboard instead of hardcoding
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# Define the schema strictly for Groq
# 1. Define the input schema
class ApiCallerInput(BaseModel):
    # This 'extra=forbid' is what generates 'additionalProperties: false' for Groq
    model_config = ConfigDict(extra='forbid') 
    
    url: str = Field(..., description="The full URL of the API endpoint")
    method: str = Field(..., description="The HTTP method (GET, POST, PUT, DELETE)")
    headers: dict = Field(default_factory=dict, description="HTTP headers as a dictionary")
    json_body: dict = Field(default_factory=dict, description="The JSON request body as a dictionary")

# 2. Define the tool as a CLASS (Do NOT use @tool decorator here)
class ApiCallerTool(BaseTool):
    name: str = "api_caller_tool"
    description: str = "Executes a real REST API call (GET, POST, PUT, DELETE)."
    args_schema: type[BaseModel] = ApiCallerInput

    def _run(self, url: str, method: str, headers: dict = None, json_body: dict = None) -> dict:
        try:
            # Ensure we have dicts, not None
            actual_headers = headers if isinstance(headers, dict) else {}
            actual_body = json_body if isinstance(json_body, dict) else {}
            
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=actual_headers,
                json=actual_body if method.upper() in ["POST", "PUT", "PATCH"] else None,
                timeout=15
            )
            return {
                "status_code": response.status_code,
                "body": response.text[:2000], # Truncate to save context
                "headers": dict(response.headers)
            }
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

# 3. Instantiate the tool once
api_caller_tool = ApiCallerTool()
# @CrewBase
# class ApiTestingCrew():
#     # Use the stable 2026 Gemini model
#     gemini_llm = LLM(
#         model="gemini/gemini-2.5-flash", 
#         api_key=os.getenv("GEMINI_API_KEY")
#     )
@CrewBase
class ApiTestingCrew():
    # UPDATED: Using Groq LLM instead of Gemini 
    # Options: "groq/llama-3.3-70b-versatile" or "groq/mixtral-8x7b-32768"
    agents_config = 'agents.yaml'  # Removed 'config/' prefix
    tasks_config = 'tasks.yaml'    # Removed 'config/' prefix
    groq_llm = LLM(
        model="groq/llama-3.3-70b-versatile", 
        #api_key=os.getenv("GROQ_API_KEY")
        api_key=groq_api_key      
    )

    @agent
    def api_requirement_collector(self) -> Agent:
        return Agent(
            config=self.agents_config['api_requirement_collector'],
            #llm=self.gemini_llm,
            llm=self.groq_llm,
            verbose=True
        )

    @agent
    def api_executor(self) -> Agent:
        return Agent(
            config=self.agents_config['api_executor'],
            tools=[api_caller_tool], # The agent uses our Python tool
           # llm=self.gemini_llm,            
           llm=self.groq_llm,
            allow_delegation=False,
            verbose=True
        )

    @agent
    def test_result_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['test_result_analyst'],
           # llm=self.gemini_llm,
           llm=self.groq_llm,
            verbose=True
        )
    
    # Define tasks using the configs...
    @task
    def collection_task(self) -> Task:
        return Task(config=self.tasks_config['input_collection_task'],
                    agent=self.api_requirement_collector())
                     

    @task
    def execution_task(self) -> Task:
        return Task(config=self.tasks_config['api_execution_task'],
                    agent=self.api_executor())
        

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config['reporting_task'],
                    agent=self.test_result_analyst(),
                    output_file='api_report.md')

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            max_rpm=10, # Keep it slow for Gemini Free Tier
            verbose=True

        )











