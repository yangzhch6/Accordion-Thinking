from verl.utils import hf_processor, hf_tokenizer
tokenizer = hf_tokenizer("/mnt/weka/home/yongxin.wang/workspace/lark/swift-pipeline/ckpt/think-step/Qwen3-4B-Base-Openr1-Prompt2-Step-Fold/v0-20251222-171747/checkpoint-5493", trust_remote_code=True)
# step_token_id = tokenizer.vocab["</step_bed619fva643c0v108hd53gcy>"]
# print(f"### Using step_token_id: {step_token_id} for generation stopping criteria.")

# # print decode [32313, 11]
# print(f"### Decoding test: {tokenizer.decode([32313, 11])}")

# tokenize the str to ids int list
# test_str = "\n\n...TLDR...\n\n"
# test_token_ids = tokenizer.encode(test_str, add_special_tokens=False)
# print(f"### Tokenize test str: {test_str} to ids: {test_token_ids}")

# test_tokens = tokenizer.tokenize(test_str, add_special_tokens=False)
# print(f"### Tokenize test str: {test_str} to tokens: {test_tokens}")

# V = len(tokenizer)
# print(V)
# print(tokenizer.vocab_size)
# print(len(tokenizer.vocab))

print(list(range(len(tokenizer.vocab))))

# decode token id: 151670
print(tokenizer.decode([151669]))  # should be </step_bed619fva643c0v108hd53gcy>
print(tokenizer.decode([151670]))  # should be </step_bed619fva643c0v108hd53gcy>

