import subprocess
import multiprocessing
import logging
from os import path, getenv

CPP_PATH = "/Users/luungoc2005/Documents/Samples/llama.cpp/main"
LLAMA_PATH = "/Users/luungoc2005/Documents/Samples/llama-cpu/LLaMA-model/7B/ggml-model-q4_0.bin"

def llama_generate(prompt, *args, **kwargs):
    output = subprocess.check_output([
        CPP_PATH,
        "-m", LLAMA_PATH,
        "-p", prompt,
        "-n", "512",
        "-t", str(multiprocessing.cpu_count() - 2),
        "--temp", str(0.7),
        "--top_k", str(40),
        "--top_p", str(0.73),
        "--repeat_last_n", str(64),
        "--repeat_penalty", str(1.3),
        "-q",
    ], encoding='utf-8')
    logging.debug("Raw LLAMA output: " + output)
    return [output]