import shutil
import os
import json
import random

from transformers import LlamaForCausalLM, GenerationConfig, AutoTokenizer
import torch
import random

from llm_study.model_patch.llama3_hook import enable_capture, get_step

# customize
MODEL = "/home/yexuming/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"
output_dir = "./results"

# sample one from LongBench datasets
with open('./triviaqa.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]
sample = random.choice(samples)

model = LlamaForCausalLM.from_pretrained(
    MODEL, dtype="float16", attn_implementation="flash_attention_2").to("cuda:0")

# clean output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# patch
enable_capture(model, output_dir, capture_attn_weight=False)

# prepare prompt
tokenizer = AutoTokenizer.from_pretrained(MODEL)
prompt = f"Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{sample['context']}\n\n{sample['input']}"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
num_input_tokens = inputs['input_ids'].shape[-1]
print("Number of tokens in inputs:", num_input_tokens)

# generation
with torch.inference_mode():
    # generate with greedy search
    outputs = model.generate(**inputs, generation_config=GenerationConfig(num_beams=1,
                                                                        do_sample=False, max_new_length=128))

print(f"Total tokens: {outputs.shape[-1]}")
print(f"Total Steps: {get_step()}")
