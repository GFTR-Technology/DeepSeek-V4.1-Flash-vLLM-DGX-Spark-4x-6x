#!/bin/bash
# dl-ds41.sh: background download of DeepSeek-V4.1-Flash into /var/tmp/models (Reddie)
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -z "$HF_TOKEN" ] && [ -f ~/.cache/huggingface/token ]; then export HF_TOKEN=$(cat ~/.cache/huggingface/token); fi
D=/var/tmp/models/DeepSeek-V4.1-Flash
mkdir -p "$D"
if pgrep -f "[h]f download deepseek-ai/DeepSeek-V4.1-Flash" >/dev/null; then echo already-running; exit 0; fi
setsid nohup ~/.local/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir "$D" --max-workers 8 > ~/dl-DeepSeek-V4.1-Flash.log 2>&1 < /dev/null &
echo "started pid=$!"
