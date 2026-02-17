import os
import json
import pandas as pd
from pypdf import PdfReader

# Disable telemetry
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

from crew import ApiTestingCrew
from reporting_utils import create_allure_result


# =========================================================
# 🔧 HELPER: Normalize body for Groq tool schema
# Converts nested dict/list → JSON string
# =========================================================
def normalize_body_for_llm(body):
    """
    Ensures body is always a STRING-safe representation
    for strict tool schemas (Groq-compatible).
    """
    if body is None or body == "":
        return ""

    # If already a string (Excel text / JSON string)
    if isinstance(body, str):
        return body.strip()

    # If dict or list → stringify
    if isinstance(body, (dict, list)):
        return json.dumps(body)

    # Fallback
    return str(body)


# =========================================================
# 🚀 SINGLE REQUEST EXECUTION
# =========================================================
def run_single_request(api_details_string):
    """Executes the Crew and sends the final string to Allure"""
    inputs = {"user_query": api_details_string}

    try:
        # Run the AI Crew
        result = ApiTestingCrew().crew().kickoff(inputs=inputs)

        # Create Allure result
        create_allure_result(
            service_name=api_details_string[:40],
            agent_output=result.raw,
            status_code=200,
        )

        return result

    except Exception as e:
        create_allure_result("Crew Failure", str(e), status_code=500)
        raise e


# =========================================================
# 📦 BULK EXECUTION (Excel / CSV / PDF)
# =========================================================
def run_bulk_from_file(file_path):
    all_requests = []

    # ---------- Excel / CSV ----------
    if file_path.endswith(".xlsx") or file_path.endswith(".csv"):
        df = (
            pd.read_excel(file_path)
            if file_path.endswith(".xlsx")
            else pd.read_csv(file_path)
        )

        for _, row in df.iterrows():
            body_value = None

            # Safely parse BODY column if present
            if "body" in row and not pd.isna(row["body"]):
                try:
                    body_value = json.loads(row["body"])
                except Exception:
                    body_value = row["body"]

            normalized_body = normalize_body_for_llm(body_value)

            details = (
                f"URL: {row['url']}, "
                f"Method: {row['method']}, "
                f"Body: {normalized_body}"
            )

            all_requests.append(details)

    # ---------- PDF ----------
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_requests.append(text)

    else:
        raise ValueError("Unsupported file format")

    print(f"\n## 🚀 Starting Execution for {len(all_requests)} services\n")

    for i, request_data in enumerate(all_requests, 1):
        try:
            print(f"▶ Executing service {i}")
            run_single_request(request_data)
        except Exception as e:
            print(f"❌ Skipping service {i} due to error: {e}")


# =========================================================
# 🏁 ENTRY POINT
# =========================================================
if __name__ == "__main__":
    file_to_process = "microservices_list.xlsx"

    if os.path.exists(file_to_process):
        run_bulk_from_file(file_to_process)
    else:
        print("Excel not found. Running single test...\n")
        run_single_request(
            "URL: https://jsonplaceholder.typicode.com/posts, Method: POST"
        )
