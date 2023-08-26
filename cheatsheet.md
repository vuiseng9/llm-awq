### VSCODE config cheatsheet

```json
        {
            "name": "Evaluate (fakequantize) - llama-7b-w4-g128",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "5",
            },
            "module": "awq.entry",
            "args": [
                "--tasks", "wikitext",
                "--model_path", "/data4/vchua/hf-model/llama-7b-hf",
                "--w_bit", "4", "--q_group_size", "128",
                "--q_backend", "fake",
                "--load_awq", 
                    "/data4/vchua/hf-model/awq_cache/llama-7b-w4-g128.pt",
            ],
            "justMyCode": false
        },
```

```json
        {
            "name": "Generate and save packed AWQ Weight",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "5",
            },
            "module": "awq.entry",
            "args": [
                "--q_backend", "real",
                "--w_bit", "4", "--q_group_size", "128",
                "--model_path", "/data4/vchua/hf-model/llama-7b-hf",
                "--load_awq", "/data4/vchua/hf-model/awq_cache/llama-7b-w4-g128.pt",
                "--dump_quant", "/tmp/awq_cache/realblob-llama-7b-w4-g128.pt",
            ],
            "justMyCode": false
        },
```

```json
        {
            "name": "Evaluate with Real AWQ kernel - llama-7b-w4-g128",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "5",
            },
            "module": "awq.entry",
            "args": [
                "--tasks", "wikitext",
                "--model_path", "/data4/vchua/hf-model/llama-7b-hf",
                "--w_bit", "4", "--q_group_size", "128",
                "--load_quant", 
                    "/data4/vchua/hf-model/awq_cache/realblob-llama-7b-w4-g128.pt",
            ],
            // "--run_awq", 
            "justMyCode": false,
            "stopOnEntry": false,
        },
```

### NEW - just add --torch_awq_kernel
```json
        {
            "name": "Evaluate with Real AWQ kernel - llama-7b-w4-g128",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "5",
            },
            "module": "awq.entry",
            "args": [
                "--tasks", "wikitext",
                "--model_path", "/data4/vchua/hf-model/llama-7b-hf",
                "--w_bit", "4", "--q_group_size", "128",
                "--load_quant", 
                    "/data4/vchua/hf-model/awq_cache/realblob-llama-7b-w4-g128.pt",
                "--torch_awq_kernel"
            ],
            "justMyCode": false,
            "stopOnEntry": false,
        },
```
