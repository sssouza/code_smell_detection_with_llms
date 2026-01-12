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
        You are analyzing the following Java file for symptoms that may indicate the "Refused Bequest" code smell.
        Refused Bequest occurs when a subclass inherits from a parent class but does not meaningfully use, override, or specialize inherited (especially protected) members, instead focusing on unrelated new functionality and fields. This suggests the subclass may not honor or make use of the parent’s contract or responsibilities.
        Since you only have access to this file, look for local patterns and symptoms—even if you cannot fully confirm the smell.

        Please answer the following questions step by step:

        1. Inheritance Pattern:  
        Does this file define a class that extends another class? If so, what is the parent class’s name?

        2. Use of Inherited Functionality:  
        Does the subclass override, call, or make substantial use of inherited methods or fields from the parent class (e.g., method overrides that change core behavior, use of `super.`, or interacting directly with inherited state)? Are any overrides minor or trivial (e.g., calling only `super.method()` or adding a one-liner)?

        3. New/Independent Functionality:  
        Does the subclass introduce its own fields and methods that represent significant new or different responsibilities, unrelated to the parent’s likely concerns?

        4. Breadth of Subclass:  
        Is the subclass non-trivial, with several additional fields and methods, indicating it is not simply a marker or light extension?

        5. Local Symptom Summary:  
        Considering your answers above, does the subclass show symptoms of Refused Bequest, meaning it extends a parent but focuses largely on different domains, rarely or weakly uses inherited features, and introduces functionality of its own? If so, briefly state the clearest sign (e.g., “many new fields/methods; few meaningful overrides; unrelated logic dominates”).

        **Instructions:**  
        If you find symptoms or strong suspicion of Refused Bequest, answer with "YES, I found Refused Bequest" and state the main evidence in a short phrase or sentence.  
        If not, answer with "NO, I did not find Refused Bequest".
        
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
    CSV_OUTPUT   = "YOUR_DESTINY_FOLDER/gpt-5-mini_analysis_RB.csv"

    analyze_java_files_to_csv(
        source_dir=SOURCE_DIR,
        csv_output_path=CSV_OUTPUT
    )