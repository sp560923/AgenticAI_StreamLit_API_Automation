import os
import json
import uuid
import re

def create_allure_result(service_name, agent_output, status_code=200):
    results_dir = "allure-results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    test_uuid = str(uuid.uuid4())
    
    # --- ROBUST EXTRACTION ---
    def extract(pattern, text, default="N/A"):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default

    # 1. API Call Details
    url = extract(r"\* \*\*URL\*\*: (.*)", agent_output)
    method = extract(r"\* \*\*Method\*\*: (.*)", agent_output)
    content_type = extract(r"Content-type: (.*)", agent_output, default="application/json")
    payload = extract(r"\* \*\*Payload\*\*: (.*)", agent_output, default="{}")
    
    # 2. Execution Results
    summary = extract(r"\* \*\*Summary\*\*: (.*)", agent_output)
    resp_time = extract(r"\* \*\*Response Time\*\*: (.*)", agent_output, default="N/A")
    schema_match = extract(r"Schema Validation\*\*: (.*)", agent_output, default="Passed")

    # 3. JSON EXTRACTION (The Fix)
    # We look for ANY code block if 'json' specific one isn't found
    json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", agent_output)
    response_body = json_block.group(1).strip() if json_block else "{'info': 'No JSON block detected in agent output'}"

    result = {
        "uuid": test_uuid,
        "historyId": f"{service_name}-{test_uuid}",
        "name": f"API Test: {service_name}",
        "status": "passed" if status_code == 200 else "failed",
        "steps": [
            {
                "name": "API Call Details",
                "status": "passed",
                "steps": [
                    {"name": f"URL: {url}", "status": "passed"},
                    {"name": f"Method: {method}", "status": "passed"},
                    {"name": f"Content-type: {content_type}", "status": "passed"},
                    {"name": f"Request Payload: {payload}", "status": "passed"}
                ]
            },
            {
                "name": "Execution Results",
                "status": "passed",
                "steps": [
                    {"name": f"Status Code: {status_code}", "status": "passed"},
                    {"name": f"Status: {'Passed' if status_code == 200 else 'Failed'}", "status": "passed"},
                    {"name": f"Response Time: {resp_time}", "status": "passed"},
                    {"name": f"Schema Match: {schema_match}", "status": "passed"},
                    {"name": f"Summary: {summary}", "status": "passed"}
                ],
                "attachments": [
                    {
                        "name": "Raw JSON Output",
                        "type": "application/json",
                        "source": f"{test_uuid}-response.json"
                    }
                ]
            }
        ]
    }

    # Write the result file
    with open(os.path.join(results_dir, f"{test_uuid}-result.json"), 'w') as f:
        json.dump(result, f, indent=4)

    # Write the attachment file (This is what Allure displays)
    with open(os.path.join(results_dir, f"{test_uuid}-response.json"), 'w') as f:
        f.write(response_body)

    print(f"✅ Success: Allure result and JSON attachment written for {service_name}")