import pandas as pd
import ollama
prompts = pd.read_csv("dataset/prompts.csv")
model = "llama3.2"
responses = []
for index, row in prompts.iterrows():

    prompt = row["PROMPT"]

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = response["message"]["content"]

    with open("dataset/conversation_history.txt", "a", encoding="utf-8") as file:
     file.write(f"PROMPT_ID: {row['PROMPT_ID']}\n")
     file.write(f"DOMAIN: {row['DOMAIN']}\n")
     file.write(f"PROMPT: {prompt}\n")
     file.write(f"RESPONSE: {answer}\n")
     file.write("-" * 80 + "\n")

    responses.append({
        "PROMPT_ID": row["PROMPT_ID"],
        "DOMAIN": row["DOMAIN"],
        "PROMPT": prompt,
        "RESPONSE": answer
    })

result = pd.DataFrame(responses)

result.to_csv(
    "dataset/llm_responses.csv",
    index=False
)

comparison = result[["PROMPT_ID", "DOMAIN", "PROMPT", "RESPONSE"]]

comparison.to_csv(
    "dataset/response_comparison.csv",
    index=False
)

print("Responses saved successfully.")
print("Response comparison saved successfully.")
print("Conversation history saved successfully.")