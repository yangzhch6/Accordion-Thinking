from pathlib import Path
import random

import pandas as pd

SYS_PROMPT_THINK = """Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. 

In the Thought section, present your reasoning using the format:“<think>\n {thoughts} </think>\n”. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. You should conduct coarse-grained step reasoning, and insert a summary after each step within <step_bed619fva643c0v108hd53gcy></step_bed619fva643c0v108hd53gcy> tags. 

After “</think>\n” in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section.

If applicable, include the Answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions."""

BOXED_CHOICE_SUFFIX = (
    "\nYour final answer should be a boxed answer include your final choice, such as: \\boxed{A}."
)

INPUT_FILES = [
    ("arc_c", Path("scripts/eval/ood/valid.arc_c.parquet")),
    ("gpqa", Path("scripts/eval/ood/valid.gpqa.parquet")),
]
OUTPUT_FILE = Path("data/think-fold/ood.parquet")
DEBUG_OUTPUT_FILE = Path("data/think-fold/ood_debug.parquet")


def get_user_content(prompt):
    for message in prompt:
        if message["role"] == "user":
            return message["content"].rstrip()
    raise ValueError("No user message found in prompt.")


def format_example(row, data_source, index):
    ground_truth = row["reward_model"]["ground_truth"].strip()
    split = row.get("extra_info", {}).get("split", "default")
    user_content = get_user_content(row["prompt"]) + BOXED_CHOICE_SUFFIX

    return {
        "prompt": [
            {"role": "system", "content": SYS_PROMPT_THINK},
            {"role": "user", "content": user_content},
        ],
        "data_source": data_source,
        "reward_key": "MATH-500",
        "ability": row.get("ability", "math"),
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": {
            "solution": f"\\boxed{{{ground_truth}}}",
            "index": index,
            "split": split,
        },
    }


def main():
    formatted_dataset = []

    for data_source, input_file in INPUT_FILES:
        rows = pd.read_parquet(input_file).to_dict("records")
        for row in rows:
            formatted_dataset.append(
                format_example(row=row, data_source=data_source, index=len(formatted_dataset))
            )

    print(formatted_dataset[0])
    print(f"Total samples: {len(formatted_dataset)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(formatted_dataset)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    random.seed(42)
    debug_samples = random.sample(formatted_dataset, 8)
    debug_df = pd.DataFrame(debug_samples)
    debug_df.to_parquet(DEBUG_OUTPUT_FILE, index=False)
    print(f"Saved to {DEBUG_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
