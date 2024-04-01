from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "tokyotech-llm/Swallow-MS-7b-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

model = AutoModelForCausalLM.from_pretrained(model_name)
prompt = "東京工業大学の主なキャンパスは、"
input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)
tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
    temperature=0.99,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)