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
        You are analyzing the following Java file for symptoms that may indicate the "Data Class" code smell.
        A Data Class is a class that mainly holds data, provides little or no functionality, and is primarily composed of fields and simple accessors (getters/setters), rather than meaningful behavior.
        Since you only have access to this file, focus on local patterns and structures that could contribute to this smell.

        Please answer the following questions step by step:

        1. Many Fields:
        Does this file define a class with many fields (attributes), especially public fields or fields with public accessors?

        2. Accessor Methods:
        Does the class provide many simple accessor methods (getters and/or setters) for its fields?

        3. Lack of Functional Methods:
        Does the class have few or no methods that implement meaningful behavior (i.e., most methods are just accessors or trivial)?

        4. Encapsulation:
        Does the class expose its data directly (public fields) or indirectly (many public accessors), rather than hiding it behind meaningful services?

        5. Summary Judgment:
        Based on your analysis, does this file define a class that is primarily a data holder with little or no meaningful behavior (i.e., a Data Class)?

        Instructions:
        Please start your answer with "YES, I found Data Class" if you detect symptoms that could indicate this smell, or "NO, I did not find Data Class" if you do not. Do not explain your reasoning in detail,just answer the questions and provide the summary as instructed.
        
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
    CSV_OUTPUT   = "YOUR_DESTINY_FOLDER/gpt-5-mini_analysis_DC.csv"

    analyze_java_files_to_csv(
        source_dir=SOURCE_DIR,
        csv_output_path=CSV_OUTPUT
    )