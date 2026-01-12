import os
import csv
from huggingface_hub import InferenceClient
from typing import List

def analyze_java_files_to_csv_hf(
    source_dir: str,
    csv_output_path: str,
    hf_model: str,
    hf_token_env: str = "HF_API_TOKEN",
    system_prompt: str = "You are a helpful assistant specialized in Java code smell detection.",
    user_template: str = """
        You are analyzing the following Java file for symptoms that may indicate the "Shotgun Surgery" code smell.
        Shotgun Surgery occurs when a small change in requirements would require making many small, similar changes in multiple places across the codebase.
        Since you only have access to this file, focus on local patterns and structures that could contribute to this smell if they are present in other files as well.

        Please answer the following questions step by step:

        1. Repetitive or Similar Methods:
        Does this file contain many methods that perform similar, small, or repetitive tasks (such as forwarding calls, updating similar fields, or making similar changes)?

        2. Duplicated or Repeated Logic:
        Are there patterns where the same or very similar logic is repeated across multiple methods or classes within this file?

        3. Widespread External Interactions:
        Do methods in this file frequently interact with many different classes, especially in a way that suggests a change in one place would require similar changes in many methods?

        4. Change Impact Within the File:
        If a business rule, data structure, or method signature used in this file were to change, would it likely require updating many methods in this file? Please give examples.

        5. Summary Judgment:
        Based on your analysis, does this file show signs that a small change in requirements would require many small edits in multiple places (i.e., Shotgun Surgery)? If so, summarize the evidence.

        Instructions:
        Please start your answer with "YES, I found Shotgun Surgery" if you detect symptoms that could indicate this smell, or "NO, I did not find Shotgun Surgery" if you do not. Do not explain your reasoning in detail, just answer the questions and provide the summary as instructed.
    
        ```java{code}```
        """,
    temperature: float = 1.0,
    max_tokens: int = 1500,
):
    """
    For each .java file under source_dir, calls a HuggingFace Inference API
    via InferenceClient.chat.completions.create(...) and writes the results
    (relative_path, analysis) to a CSV file.
    """
    # 1. Get Hugging Face API token from environment variables
    hf_token = "YOUR_TOKEN_HERE"
    if not hf_token:
        raise RuntimeError(
            f"Please set the '{hf_token_env}' environment variable with your "
            f"Hugging Face API token."
        )

    # 2. Initialize the Inference Client
    client = InferenceClient(token=hf_token)

    # 3. Gather all .java files to be analyzed
    java_files = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if fname.endswith(".java"):
                full_path = os.path.join(root, fname)
                # Get path relative to the source dir for cleaner logs
                rel_path = os.path.relpath(full_path, source_dir)
                java_files.append((full_path, rel_path))

    print(f"Found {len(java_files)} Java files to analyze in '{source_dir}'.")

    # 4. Prepare the output CSV file
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    with open(
        csv_output_path, "w", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(["file_path", "analysis"])

        # 5. Loop through each file, call the API, and write the result
        for idx, (full_path, rel_path) in enumerate(java_files, 1):
            print(f"[{idx}/{len(java_files)}] Analyzing: {rel_path}")
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                # If a file can't be read, log the error and continue
                analysis = f"[ERROR] Could not read file: {e}"
                writer.writerow([rel_path, analysis])
                continue

            # Prepare the messages for the chat model
            user_prompt = user_template.format(code=code)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Call the Hugging Face API
            try:
                response = client.chat.completions.create(
                    model=hf_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Extract the text content from the response
                analysis = response.choices[0].message.content.strip()
            except Exception as e:
                # If the API call fails, log the error
                analysis = f"[ERROR] API call failed: {e}"

            # Write the file path and the model's analysis to the CSV
            writer.writerow([rel_path, analysis])

    print(
        f"\nAnalysis complete. All results have been saved to '{csv_output_path}'."
    )


if __name__ == "__main__":
    # === User Configuration ===
    # The folder containing the Java source files you want to analyze.
    SOURCE_DIR = r"YOUR_ORIGIN_FOLDER"

    # The full path where the output CSV file will be saved.
    CSV_OUTPUT = r"YOUR_DESTINY_FOLDER/deepseekR1_analysis_SS.csv"

    # The repository ID of the chat model on Hugging Face Hub.
    # Example: "mistralai/Mistral-7B-Instruct-v0.2"
    HF_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   

    # The name of the environment variable holding your Hugging Face token.
    HF_TOKEN_ENV = "HF_API_TOKEN"
    # ==========================

    analyze_java_files_to_csv_hf(
        source_dir=SOURCE_DIR,
        csv_output_path=CSV_OUTPUT,
        hf_model=HF_MODEL,
        hf_token_env=HF_TOKEN_ENV,
    )