#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import transformers
from transformers import TFAutoModelForCausalLM, AutoTokenizer

model_name = "tiiuae/falcon-7b-instruct"
model = TFAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100, temperature=0.7)
