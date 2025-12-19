import shutil
import os
import json
import random

from transformers import LlamaForCausalLM, GenerationConfig, AutoTokenizer
import torch

from llm_study.model_patch.llama3_hook import enable_capture, get_step

# customize
jsonl_path = "/home/yexuming/datasets/fwe-ruler-48k.jsonl"
MODEL = "/home/yexuming/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"
output_dir = "./results"


def sample_from_jsonl(jsonl_path, sample_num=1):
    """从指定路径解析 .jsonl 数据集文件，并随机采样一条样本"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sample_lines = random.sample(lines, sample_num)
    samples = [json.loads(line) for line in sample_lines]
    return samples


model = LlamaForCausalLM.from_pretrained(
    MODEL, dtype="float16", attn_implementation="flash_attention_2").to("cuda:0")

# clean output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# patch
enable_capture(model, output_dir, capture_attn_weight=False)

# generate inputs
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# sample one from fwe datasets
sample = sample_from_jsonl(jsonl_path, 1)[0]

prompt = sample["input"]
expected_outputs = sample["outputs"]

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
num_input_tokens = inputs['input_ids'].shape[1]
print("Number of tokens in inputs:", num_input_tokens)


with torch.inference_mode():
    # generate with greedy search
    outputs = model.generate(**inputs, generation_config=GenerationConfig(num_beams=1,
                                                                        do_sample=False, max_new_length=128))

print("output: ", tokenizer.decode(outputs[0], skip_special_tokens=True))
print("expected_outputs: ", expected_outputs)
print(f"Total Steps: {get_step()}")
