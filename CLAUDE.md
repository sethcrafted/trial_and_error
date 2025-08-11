## Overview

This repository is for exploration and research of LLMs. It  is presently in early stages. The repository is organized as follows:

## Organization 

```
experimental/
- Contains early investigations and true "trial and error" type work to see how things work together. It's meant for exploring 
  tutorials, sketching out ideas, or just seeing if something works

llm_env/
- This directory can be ignored, it's for the python virtual environment setup to make sure I have the right things running. 

models/
- This directory contains different LLM models, primarily from huggingface. The top level has different script to search, download, and verify
  models are working as expected. 
models/local/
- This is the directory which actually contains downloaded models. Claude doesn't need to read this directory unless explicitly asked to, 
  since it contains a lot of bloat from trusted sources. 
```

## Goals
- Setup and run multiple different LLM models
- Understanding Inference optimization, including curriculum learning understanding for fine-tuning
- Exploring mechanistic interpretability
- Building visualization tools to understand model architecture
- Leveraging RL to optimize different parts of the ML process
- Setting up basic MCP to get familiar with how it might work

## Stage 1: Get Modelings working

We're currently at this stage, we're were getting multiple models setup and working on differfent hardware. At present, this repository is used 
on two different machines:
- An M1 Pro Mac 
- An AMD powered Mini-PC with 64 GB RAM and a Radeon iGPU]
```
