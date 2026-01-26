import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoTokenizer

# 使用自定义模型类，避免 AutoModel 无法识别 GraphLlama
from graphgpt.gr.graphgpt.model import GraphLlamaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", required=True, help="Path to base model (Stage1 checkpoint)")
    parser.add_argument("--lora_adapter", required=True, help="Path to LoRA adapter (Stage2 checkpoint)")
    parser.add_argument("--output_dir", required=True, help="Directory to save merged model")
    parser.add_argument("--projector_path", default=None, help="Optional path to graph_projector.bin to copy alongside")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="dtype for loading base model")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model from {args.base_model} with dtype={torch_dtype}...")
    base_model = GraphLlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
    )

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    print(f"Loading LoRA adapter from {args.lora_adapter} on CPU and merging...")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter, device_map={"": "cpu"})
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.projector_path and os.path.exists(args.projector_path):
        target_proj = os.path.join(args.output_dir, os.path.basename(args.projector_path))
        shutil.copy2(args.projector_path, target_proj)
        print(f"Copied projector to {target_proj}")

    print("Done. Use --model-name pointing to the merged directory for inference.")


if __name__ == "__main__":
    main()
