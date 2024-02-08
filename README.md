# LLMViewer

Explore the intricate facets of Large Language Models (LLMs) with LLMViewer, a powerful tool designed to provide comprehensive insights into key aspects of model inference. Uncover details such as computational load, data volume, transmission metrics, and hardware roofline model analysis. Whether you're delving into the complexities of computation, assessing various models, evaluating data transmission, or examining the theoretical hardware efficiency with the roofline model, LLMViewer offers a user-friendly interface for an in-depth exploration of these critical components in large language model inference. Elevate your understanding of LLMs with this tool.

# example

```bash
python3 analyze_cli.py facebook/opt-125m nvidia_A6000
python3 analyze_cli.py meta-llama/Llama-2-7b-hf nvidia_A6000 --batchsize 1 --seqlen 2048
python3 analyze_cli.py THUDM/chatglm3-6b nvidia_V100 --seqlen 2048

python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 1 --seqlen 2048

python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 1 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 16 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 1 --seqlen 8192

```
