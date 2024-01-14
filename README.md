# LLMViewer

Explore the intricate facets of Large Language Models (LLMs) with LLMViewer, a powerful tool designed to provide comprehensive insights into key aspects of model inference. Uncover details such as computational load, data volume, transmission metrics, and hardware roofline model analysis. Whether you're delving into the complexities of computation, assessing various models, evaluating data transmission, or examining the theoretical hardware efficiency with the roofline model, LLMViewer offers a user-friendly interface for an in-depth exploration of these critical components in large language model inference. Elevate your understanding of LLMs with this tool.

# example

```bash
python3 model_viewer.py facebook/opt-125m configs/opt.py
python3 model_viewer.py meta-llama/Llama-2-7b-hf configs/llama-2.py --batchsize 2
python3 model_viewer.py THUDM/chatglm3-6b configs/chatglm3.py --seqlen 2048

python3 model_viewer.py meta-llama/Llama-2-13b-hf configs/llama-2.py --batchsize 1 --seqlen 2048

```
