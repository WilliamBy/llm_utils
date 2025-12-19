from transformers import AutoTokenizer
from llm_study.model_patch.llama3_hook import enable_capture, get_step

from transformers import LlamaForCausalLM, GenerationConfig

# custom
MODEL = "/home/yexuming/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"
output_dir = "./results"

# clean output directory
import os
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# load model
model = LlamaForCausalLM.from_pretrained(
    MODEL, dtype="float16", attn_implementation="eager").to("cuda:0")

# capture qkv and attention weight and RoPE QK
enable_capture(model, output_dir, mode="qkvaQK")

# sample from dataset
from random import choice
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('AI-ModelScope/aime_2024 ', split='train')
sample = choice(ds)
print(f"Sample: \n{sample}")

# generate inputs
tokenizer = AutoTokenizer.from_pretrained(MODEL)
template = "Problem: %s\nSolution:"
input_text = template % sample["Problem"]
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
num_input_tokens = inputs['input_ids'].shape[1]
print("Number of tokens in inputs:", num_input_tokens)


# generate with greedy search
outputs = model.generate(
    **inputs,
    num_beams=1,
    do_sample=False,
    max_new_tokens=128,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
num_output_tokens = outputs.shape[1]
num_new_tokens = num_output_tokens - num_input_tokens
print("Generated new tokens: ", num_new_tokens)
print(f"Total Steps: {get_step()}")

# output
answer = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
print("Solution：", answer)


