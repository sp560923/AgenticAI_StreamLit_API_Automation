import os
import requests
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

# --- UPDATED CUSTOM TOOL ---
@tool("api_caller_tool")
def api_caller_tool(url: str, method: str, headers: dict = {}, json_body: dict = {}):
    """Executes a real REST API call (GET, POST, PUT, DELETE)."""
    try:
        # If the LLM passes None, default back to empty dict
        actual_headers = headers if headers is not None else {}
        actual_body = json_body if json_body is not None else {}
        
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=actual_headers,
            json=actual_body if method.upper() in ["POST", "PUT", "PATCH"] else None,
            timeout=10
        )
        return {
            "status_code": response.status_code,
            "body": response.text,
            "headers": dict(response.headers)
        }
    except Exception as e:
        return f"Request failed: {str(e)}"
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
    groq_llm = LLM(
        model="groq/llama-3.3-70b-versatile", 
        api_key=os.getenv("GROQ_API_KEY")
       #api_key="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"
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