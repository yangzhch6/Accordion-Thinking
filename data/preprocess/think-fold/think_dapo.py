import pandas as pd
from datasets import concatenate_datasets, load_dataset

SYS_PROMPT_THINK = """Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. 

In the Thought section, present your reasoning using the format:“<think>\n {thoughts} </think>\n”. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. You should conduct coarse-grained step reasoning, and insert a summary after each step within <step_bed619fva643c0v108hd53gcy></step_bed619fva643c0v108hd53gcy> tags. 

After “</think>\n” in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section.

If applicable, include the Answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions."""

# load data from huggingface open-r1/DAPO-Math-17k-Processed
train_dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", split="train")

formated_dataset = []
for line in train_dataset:
    question = line['prompt']
    solution = ""
    reward_model = line['reward_model']
    reward_model["style"] = "rule"
    formated_dataset.append({
        "prompt": [{"role":"system", "content": SYS_PROMPT_THINK},{"role": "user", "content": question}],
        "data_source": "DAPO-17k",
        "reward_key": "DAPO-17k",
        "ability": 'math',
        "reward_model": reward_model,
        "extra_info": {
            "solution": solution,
            "index": len(formated_dataset),
            "split": "train"
        },
    })

print(formated_dataset[0])
print(f"Total samples: {len(formated_dataset)}")


# save as parquet
df = pd.DataFrame(formated_dataset)
df.to_parquet("data/think-fold/dapo-17k.parquet", index=False)
