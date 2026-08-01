"""
Standalone LLM inference worker for llama-cpp-python.
Runs in its own virtual environment to isolate CUDA/model dependencies.

Usage:
    python llm_worker.py <config_json_path>

Config JSON:
{
    "model_path": "C:/LLM/model.gguf",
    "system_prompt": "...",
    "user_content": "...",
    "n_ctx": 4096,
    "n_batch": 512,
    "temperature": 0.4,
    "max_tokens": -1,
    "top_p": 0.95,
    "stream": true,
    "json_mode": true,
    "stop": ["</s>", "<end_of_turn>", "<eos>"]
}

Output (stream=true):
    One JSON line per token:  {"t": "token_text"}
    Final line:               {"done": true}

Output (stream=false):
    Single line:              {"done": true, "content": "full_response"}

Errors:
    {"error": "description"}
"""
import sys
import os
import json
import gc


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: llm_worker.py <config_json_path>"}), flush=True)
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read config: {e}"}), flush=True)
        sys.exit(1)

    model_path = config.get("model_path", "")
    system_prompt = config.get("system_prompt", "")
    user_content = config.get("user_content", "")
    n_ctx = int(config.get("n_ctx", 4096))
    n_batch = int(config.get("n_batch", 512))
    temperature = float(config.get("temperature", 0.4))
    max_tokens = int(config.get("max_tokens", -1))
    top_p = float(config.get("top_p", 0.95))
    stream = bool(config.get("stream", False))
    json_mode = bool(config.get("json_mode", True))
    stop_tokens = config.get("stop", ["</s>", "<end_of_turn>", "<eos>"])

    if not model_path or not os.path.exists(model_path):
        print(json.dumps({"error": f"Model not found: {model_path}"}), flush=True)
        sys.exit(1)

    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_batch=n_batch,
            flash_attn=True,
            verbose=False
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if stream:
            kwargs["stream"] = True
            full_text = ""
            for chunk in llm.create_chat_completion(**kwargs):
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    token = delta["content"]
                    full_text += token
                    print(json.dumps({"t": token}), flush=True)
            print(json.dumps({"done": True}), flush=True)
        else:
            response = llm.create_chat_completion(**kwargs)
            content = response["choices"][0]["message"]["content"]
            print(json.dumps({"done": True, "content": content}), flush=True)

        del llm
        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
