import json

input_path = "data/QandA.txt"
output_path = "data/processed/QandA.json"

qa_list = []
question = None
answer = None

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question is not None and answer is not None:
                qa_list.append({
                    "question": question,
                    "answer": answer
                })
                question = None
                answer = None

with open(output_path, "w", encoding="utf-8") as out_f:
    json.dump(qa_list, out_f, ensure_ascii=False, indent=2)

print(f"Converted {len(qa_list)} Q/A pairs to {output_path}")