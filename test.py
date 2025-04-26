"""
Test script for the intent classifier API.
Example curl commands and Python requests to test the API endpoints.
"""
import requests
import json
import subprocess
import sys

# Update to match the default port in intent-classifier.py
BASE_URL = "http://localhost:8000"


def run_curl_command(command):
    """Execute curl command and print the result"""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Status code: {process.returncode}")
    print(f"Response: {process.stdout}")
    if process.stderr:
        print(f"Error: {process.stderr}")
    print("-" * 50)
    return process.stdout


def main():
    print("\n=== CURL COMMANDS FOR TESTING ===\n")

    # Test 1: Health check endpoint
    print("1. Testing /health endpoint")
    run_curl_command(f"curl -X GET {BASE_URL}/health")

    # Test 2: Model info endpoint
    print("2. Testing /model_info endpoint")
    run_curl_command(f"curl -X GET {BASE_URL}/model_info")

    # Test 3: Classify endpoint with different queries
    print("3. Testing /classify endpoint with different queries")

    queries = [
        "do you have the new MacBook Pro in stock",
        "what is the price of the iPhone 13",  # Removed apostrophe
        "how does the Samsung Galaxy compare to iPhone",
        "I am looking for a laptop with good battery life",  # Removed apostrophe
        "show me red shoes"
    ]

    for query in queries:
        print(f"\nClassifying: '{query}'")
        # Use json.dumps to properly escape the query string
        json_payload = json.dumps({"query": query})
        command = f'curl -X POST {BASE_URL}/classify -H "Content-Type: application/json" -d \'{json_payload}\''
        run_curl_command(command)


if __name__ == "__main__":
    main()