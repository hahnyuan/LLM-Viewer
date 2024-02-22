# LLM-Viewer

LLM-Viewer is a powerful tool for visualizing Language and Learning Models (LLMs) and analyzing the performance on different hardware platforms. It enables network-wise analysis, considering factors such as peak memory consumption and total inference time cost. With LLM-Viewer, you can gain valuable insights into LLM inference and performance optimization.


## Workflow

![LLM-Viewer Workflow](figs/workflow.svg)

As shown in the Figure, the workflow consists of the following steps:

1. Input the LLM and gather essential information about each layer, including the computation count, input and output tensor shapes, and data dependencies.
2. Provide input for the hardware and generate a roofline model that takes into account the computation capacity and memory bandwidth of the hardware.
3. Configure the inference settings, such as the batch size, prompt token length, and generation token length.
4. Configure the optimization settings, such as the quantization bitwidth, utilization of FlashAttention, decoding methods, and other system optimization techniques.
5. Use the LLM-Viewer Analyzer to analyze the performance of each layer based on the roofline model and layer information. It also tracks the memory usage of each layer and calculates the peak memory consumption based on data dependencies. The overall network performance of the LLM can be obtained by aggregating the results of all layers.
6. Generate a report that provides information such as the maximum performance and performance bottlenecks of each layer and the network, as well as the memory footprint. The report can be used to analyze curves, such as batch size-performance and sequence length-performance curves, to understand how different settings impact performance.
7. Access the LLM-Viewer web viewer for convenient visualization of the network architecture and analysis results. This tool facilitates easy configuration adjustment and provides access to various data for each layer.


## Usage

Clone the LLM-Viewer repository from GitHub: 
```bash   git clone https://github.com/hahnyuan/LLM-Viewer.git   ```

To analyze an LLM using LLM-Viewer in command line interface (cli), run the following command:

```bash
python3 analyze_cli.py facebook/opt-125m nvidia_A6000
python3 analyze_cli.py meta-llama/Llama-2-7b-hf nvidia_A6000 --batchsize 1 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 16 --seqlen 2048
python3 analyze_cli.py meta-llama/Llama-2-13b-hf nvidia_A6000 --batchsize 1 --seqlen 8192
```
