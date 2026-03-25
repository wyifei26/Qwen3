This folder provides scripts to reproduce evaluation results across various benchmarks for the **Qwen** series of large language models.

## Supported Benchmarks

Currently, we support the following benchmark:

| Model | Dataset | Config | Reproduced Score |
|-------|--------|--------|------------------|
| Qwen3-235B-A22B-Instruct-2507 | ARC-AGI 1 (pass@1) | [./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml](./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml) | 40.75 |
| Qwen3-1.7B-Base | GSM8K (8-shot CoT, greedy) | [./configs/GSM8K-Qwen3-1.7B-Base.yaml](./configs/GSM8K-Qwen3-1.7B-Base.yaml) | — |

In the meantime, you can find the model outputs and final evaluation results in the [`./output`](./output) and [`./eval_res`](./eval_res) directories, respectively.

Additional benchmarks will be added in future updates. 


## Evaluation Guide

Follow the steps below to reproduce the reported scores.

### Step 0: Prerequisites

Ensure you have:
- Python ≥ 3.9
- Either [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sgl) installed

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Start vLLM Server

Launch the vLLM inference server using the command below:

```bash
export MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"  # Replace with desired model
export MODEL_PATH="$MODEL_NAME"  # Or path to local checkpoint
export NUM_GPUS=8

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $NUM_GPUS \
    --enforce-eager \
    --port 8030
```

> 💡 Adjust `tensor_parallel_size` according to your GPU setup.

### Optional: Start SGLang Router (Recommended for Faster Evaluation)

Since evaluations can take several days, we recommend using **SGLang** with data parallelism to accelerate inference. See the [SGLang Router documentation](https://docs.sglang.ai/router/router.html) for details.

Start the SGLang router server:

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

> ⚠️ Adjust `dp_size` based on available resources, and ensure consistency in port configuration for subsequent steps.


### Step 2: Run Inference

Once the inference server is running, generate model responses using the multithreaded inference script.

```bash
mkdir -p output

# Example: Evaluate on ARC-AGI
python generate_api_answers/infer_multithread.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml
```

#### Resume Interrupted Inference

If the process is interrupted, simply re-run the same command. The script will automatically detect existing outputs and resume generation for incomplete prompts.

### Step 3: Compute Scores

After inference completes, evaluate the results using the scoring script:

```bash
mkdir -p eval_res

python eval/eval.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml \
    > eval_res/ARCAGI-Qwen3-235B-A22B-Instruct-2507_eval_result.txt
```

The final score will be saved to the specified output file.


---

## GSM8K Evaluation — Qwen3-1.7B-Base

### Prompt and Parameters

**Dataset:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test split (1,319 problems).

**Prompt:** 8-shot chain-of-thought (CoT) few-shot prompt.  Eight in-context
examples are drawn from the GSM8K training set (Cobbe et al., 2021) and
prepended to every test question.  Each example follows the format:

```
Question: <question text>
Answer: <step-by-step reasoning> #### <final numeric answer>
```

The model is expected to reproduce the same format, and the string after
`####` is taken as its predicted answer for scoring.

**Sampling parameters (greedy decoding):**

| Parameter | Value |
|-----------|-------|
| `temperature` | 0 |
| `top_p` | 1.0 |
| `top_k` | −1 (disabled) |
| `max_tokens` | 1024 |
| `presence_penalty` | 0.0 |

**Scoring:** exact-match accuracy after normalising both the predicted and
ground-truth answers to a canonical numeric string (commas stripped,
integer-valued floats collapsed, e.g. `8.0` → `8`).

### Reproducing the Score

#### Step 0: Prepare the data

```bash
cd eval
pip install datasets          # if not already installed
python data/gsm8k_prepare.py  # writes data/gsm8k.jsonl
```

#### Step 1: Start the vLLM server

```bash
export MODEL_NAME="Qwen/Qwen3-1.7B-Base"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --trust-remote-code \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --port 8030
```

#### Step 2: Run inference

```bash
mkdir -p output

python generate_api_answers/infer_multithread.py \
    --config configs/GSM8K-Qwen3-1.7B-Base.yaml
```

#### Step 3: Compute scores

```bash
mkdir -p eval_res

python eval/eval.py \
    --config configs/GSM8K-Qwen3-1.7B-Base.yaml \
    > eval_res/GSM8K-Qwen3-1.7B-Base_eval_result.txt
```
