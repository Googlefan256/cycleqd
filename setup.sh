#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install transformers[torch] git+https://github.com/EleutherAI/lm-evaluation-harness
cp cfg_placeholder.py cfg.py