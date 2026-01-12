from verl.utils import hf_processor, hf_tokenizer
tokenizer = hf_tokenizer("/mnt/weka/home/yongxin.wang/workspace/lark/swift-pipeline/ckpt/think-step/Qwen2.5-Math-7B-16k-think-Prompt2-Step-Fold/v0-20251228-174712/checkpoint-5493", trust_remote_code=True)
# step_token_id = tokenizer.vocab["</step_bed619fva643c0v108hd53gcy>"]
# print(f"### Using step_token_id: {step_token_id} for generation stopping criteria.")

# # print decode [32313, 11]
# print(f"### Decoding test: {tokenizer.decode([32313, 11])}")

# tokenize the str to ids int list
test_str = "\n\n...TLDR...\n\n"
test_token_ids = tokenizer.encode(test_str, add_special_tokens=False)
print(f"### Tokenize test str: {test_str} to ids: {test_token_ids}")

test_tokens = tokenizer.tokenize(test_str, add_special_tokens=False)
print(f"### Tokenize test str: {test_str} to tokens: {test_tokens}")