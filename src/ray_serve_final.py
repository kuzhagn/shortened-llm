from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import ray
from ray import serve
from fastapi import Request
import time

# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from peft import PeftModel, PeftConfig
from utils import get_model, set_seed
from argparse import ArgumentParser


# Initialize Ray
ray.init()
serve.start(detached=True)


# Argument parser
def build_argparser():
    parser = ArgumentParser(description="Script to evaluate perplexity with a given model")
    parser.add_argument(
        "-m", "--model",
        default="meta-llama/Llama-3.1-8B",
        help="Model identifier (Hugging Face hub path or local path). Default: %(default)s"
    )
    parser.add_argument(
        "--quantization",
        choices=["2bit", "4bit", "8bit", "none"],
        default="none",
        help="Quantization type for model loading. Choices: [2bit, 4bit, 8bit, none]. Default: %(default)s"
    )

    parser.add_argument(
        "--model_type",
        choices=["finetuned", "pretrain"],
        default="pretrain",
        help="Model type for loading. Choices: [finetuned, pretrain]. Default: %(default)s"
    )

    parser.add_argument(
        "--attn_implementation",
        choices=['flash_attention_2', 'eager'],
        default="flash_attention_2",
        help="Attention implementation. Choices: [flash_attention_2, eager]. Default: %(default)s"
    )

    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--num_output", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)

    parser.add_argument("--output_dir", type=str, default="results/llama-3.1-8b/ppl")

    parser.add_argument("--device", type=str, default="cuda", help="device")

    return parser



@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self, args):
        print("Loading model...")
        self.model, self.tokenizer, _ = get_model(
                                        base_model=args.base_model,
                                        ckpt=args.ckpt,
                                        lora_ckpt=args.lora_ckpt,
                                        tokenizer=args.tokenizer,
                                        model_type=args.model_type,
                                        device=args.device,
                                        fix_decapoda_config=args.fix_decapoda_config,
                                        use_bfloat=args.use_bfloat,
                                    )

        self.model.eval() 

        # self.tokenizer = tokenizer
        print("Model loaded.")

    async def __call__(self, request: Request):
        payload = await request.json()
        prompt = payload.get("prompt", "")
        if not prompt:
            return {"error": "Missing prompt"}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start_time = time.time()
        output_ids = self.model.generate(input_ids=inputs["input_ids"], 
                                        do_sample=True,
                                        top_k=payload.get("top_k", 50),
                                        top_p=payload.get("top_p", 0.95),
                                        temperature=payload.get("temperature", 1.0),
                                        max_new_tokens=payload.get("max_new_tokens", 128),
)

        latency = time.time() - start_time

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2

        # Measure memory used during generation
        print(f"[Generate] Allocated memory: {mem_alloc:.2f} MB")
        print(f"[Generate] Reserved memory:  {mem_reserved:.2f} MB")

        return {
            "output": output_text,
            "latency_sec": latency, 
            "memory_alloc": mem_alloc, 
            "memory_reserved":  mem_reserved
        }


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="baffo32/decapoda-research-llama-7B-hf",
        help="base model name",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "pruneLLM", "tune_pruneLLM"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--num_output", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/llama-7b-hf/ppl")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    serve.run(LLMDeployment.bind(
        args),
        route_prefix="/generate")

