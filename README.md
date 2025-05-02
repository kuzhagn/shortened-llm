Fine-Tuning and Probing LLaMA-3.1-8B (Adapted from Shortened LLaMA)

This repository includes a focused adaptation of the  [**Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods**](https://arxiv.org/abs/2402.02834) by Nota AI,  [**original Github repo**]([https://arxiv.org/abs/2402.02834](https://github.com/Nota-NetsPresso/shortened-llm.git).

The original work introduces depth pruning for large language models and compares retraining methods such as LoRA and continued pretraining (CPT).

In this adaptation, I focus exclusively on:

Probing block importance using perplexity or Taylor expansion
LoRA-based fine-tuning of the meta-llama/Llama-3.1-8B model
Zero-shot evaluation and inference latency estimation using Ray Serve
Only a small subset of the original codebase was reused and lightly modified. The rest of the pipeline, including extensive pruning experiments, is not reproduced here.

🔧 Setup

```
conda create -n llama3-pruning python=3.9
conda activate llama3-pruning
git clone https://github.com/kuzhagn/shortened-llm.git
cd shortened-llm
pip install -r requirement.txt
pip install ray[serve]
```

2. Run with Perplexity Criterion

```
bash script/prune_llama3.1-8b_crit-ppl.sh
```
4. Or Run with Taylor Expansion Criterion
```
bash script/prune_llama3.1-8b_crit-taylor.sh
```

📌 Notes
```
CUDA_VISIBLE_DEVICES is set within each script.
LoRA retraining uses: r=8, batch=64, micro_batch=4, 2 epochs.
Ray Serve hosts the model for latency measurement using REST-style HTTP calls.
Both scripts are self-contained and designed to be run independently.
```
🔗 Credits

This project is adapted from:

Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
Bo-Kyeong Kim et al., 2024 – arXiv:2402.02834




