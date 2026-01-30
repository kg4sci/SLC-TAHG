import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataname = 'SLC_database'
data = torch.load(f"../SLC_database.pt")
raw_texts = data.raw_texts
model_name = "llama2-7b-hf-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer.pad_token = tokenizer.bos_token
# 确保生成时有有效的 pad/eos 配置，避免数值异常
pad_id = tokenizer.pad_token_id
eos_id = tokenizer.eos_token_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 显式使用 float16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    torch_dtype=torch.float16,
).to(device)
index_list = list(range(len(raw_texts)))

if dataname == 'SLC_database':
    prompt = (
        "\nYou are an expert reviewer.\n"
        "Text:\n{TEXT}\n\n"
        "Question: Which relations best fits SLC node and pathway node? Building upon the relationship between the previous pair of nodes, which relations best fits pathw node and disease node?Determine the most appropriate relationship between the two pairs of nodes that share conditional dependencies."
        "\nAnswer:"
    )
batch_size = 2
data_loader = DataLoader(list(zip(raw_texts, index_list)), batch_size=batch_size, sampler=SequentialSampler(list(zip(raw_texts, index_list))))

for batch in tqdm(data_loader):
    text_batch, index_batch = batch[0], batch[1]
    batch_prompts = [prompt.format(TEXT=text) for text in text_batch]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        do_sample=False,    
        max_new_tokens=64,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # print(answers)

    os.makedirs(f"llama_response/{dataname}", exist_ok=True)
    for idx, answer in zip(index_batch, answers):
        with open(f"llama_response/{dataname}/{idx}.json", 'w') as f:
            json.dump({"answer": answer}, f)
