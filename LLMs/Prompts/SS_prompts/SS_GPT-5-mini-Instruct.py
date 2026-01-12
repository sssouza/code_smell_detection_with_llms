import os
import csv
import openai

def analyze_java_files_to_csv(
    source_dir: str,
    csv_output_path: str,
    model: str = "gpt-5-mini",
    system_prompt: str = "You are a helpful assistant specialized in Java code smell detection.",
    user_template: str = (
        """
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

        \n\n```java\n{code}\n```"""
    ),
    temperature: float = 1.0,
    max_completion_tokens: int = 1500
):
    """
    For each .java file under source_dir, calls GPT-4o via OpenAI API
    and writes (relative_path, analysis) to a CSV at csv_output_path.
    Uses the new openai>=1.0.0/chat API.
    """
    openai.api_key = "YOUR_TOKEN_HERE"
    if not openai.api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    # Gather all .java files
    java_files = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if fname.endswith(".java"):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, source_dir)
                java_files.append((full_path, rel_path))

    print(f"Discovered {len(java_files)} Java files under '{source_dir}'.")

    # Prepare output CSV
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["file_path", "analysis"])

        for idx, (full_path, rel_path) in enumerate(java_files, 1):
            print(f"[{idx}/{len(java_files)}] Analyzing {rel_path}...")
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                analysis = f"[ERROR] Could not read file: {e}"
                writer.writerow([rel_path, analysis])
                continue

            user_prompt = user_template.format(code=code)
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                analysis = response.choices[0].message.content.strip()
            except Exception as e:
                analysis = f"[ERROR] API call failed: {e}"

            writer.writerow([rel_path, analysis])

    print(f"Analysis complete. Results saved to '{csv_output_path}'.")


if __name__ == "__main__":
    SOURCE_DIR   = "YOUR_ORIGIN_FOLDER"
    CSV_OUTPUT   = "YOUR_DESTINY_FOLDER/gpt-5-mini_analysis_SS.csv"

    analyze_java_files_to_csv(
        source_dir=SOURCE_DIR,
        csv_output_path=CSV_OUTPUT
    )