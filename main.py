import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
import pandas as pd
from pypdf import PdfReader
#from AgenticAI_StreamLit_API_Automation.crew import ApiTestingCrew
from crew import ApiTestingCrew

# Import the utility from your other file
from reporting_utils import create_allure_result

def run_single_request(api_details_string):
    """Executes the Crew and sends the final string to Allure"""
    inputs = {'user_query': api_details_string}
    
    try:
        # 1. Run the AI Crew (This generates api_report.md automatically)
        result = ApiTestingCrew().crew().kickoff(inputs=inputs)
        
        # 2. Use our utility to create the Allure JSON files
        # We take the first 40 chars of the input as the name
        create_allure_result(
            service_name=api_details_string[:40], 
            agent_output=result.raw, 
            status_code=200 
        )
        
        return result
    except Exception as e:
        # Log failure if the crew crashes
        create_allure_result("Crew Failure", str(e), status_code=500)
        raise e

def run_bulk_from_file(file_path):
    all_requests = []
    # (Data extraction logic remains same: Excel/CSV/PDF)
    if file_path.endswith('.xlsx') or file_path.endswith('.csv'):
        df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
        for _, row in df.iterrows():
            details = f"URL: {row['url']}, Method: {row['method']}, Body: {row['body']}"
            all_requests.append(details)

    print(f"## Starting Execution for {len(all_requests)} services")
    for i, request_data in enumerate(all_requests, 1):
        try:
            run_single_request(request_data)
        except Exception as e:
            print(f"Skipping service {i} due to error: {e}")

if __name__ == "__main__":
    file_to_process = "microservices_list.xlsx" 
    if os.path.exists(file_to_process):
        run_bulk_from_file(file_to_process)
    else:
        print("Excel not found. Running single test...")

        run_single_request("URL: https://jsonplaceholder.typicode.com/posts, Method: POST")
