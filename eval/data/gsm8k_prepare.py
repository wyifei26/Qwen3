"""
Prepare GSM8K test data for evaluating Qwen3 base models.

This script downloads the GSM8K test split from Hugging Face, prepends the
standard 8-shot chain-of-thought (CoT) prompt used for base-model evaluation,
and writes one JSON object per line to ``gsm8k.jsonl`` in the same directory.

Usage:
    python eval/data/gsm8k_prepare.py
"""

import json
import os

# ---------------------------------------------------------------------------
# Standard 8-shot chain-of-thought prompt for GSM8K base-model evaluation.
# Examples are drawn from the GSM8K training set (Cobbe et al., 2021).
# Each example ends with "#### <number>" to signal the final numeric answer.
# ---------------------------------------------------------------------------

FEW_SHOT_PROMPT = """\
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Answer: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more on Wednesday, he had 35 - 2 = 33. #### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8

Question: {question}
Answer:"""


def prepare_gsm8k(output_path: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        )

    print("Downloading GSM8K test split from Hugging Face …")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            question = item["question"].strip()
            # Ground-truth answer is stored as "... #### <number>"
            answer_raw = item["answer"].strip()
            # Extract the numeric answer after "####"
            answer_parts = answer_raw.split("####")
            answer = answer_parts[-1].strip().replace(",", "") if len(answer_parts) > 1 else answer_raw.strip()

            prompt = FEW_SHOT_PROMPT.format(question=question)
            record = {
                "prompt": prompt,
                "answer": answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "gsm8k.jsonl")
    prepare_gsm8k(output_path)
