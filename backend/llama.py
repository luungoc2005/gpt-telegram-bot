import subprocess
import multiprocessing
import logging
from os import path, getenv

logger = logging.getLogger(__name__)
handler = logging.FileHandler('llama.log')
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

CPP_PATH = getenv('CPP_PATH')
LLAMA_PATH = getenv('LLAMA_PATH')

def llama_generate(prompt, *args, **kwargs):
    try:
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
        logger.debug("Raw LLAMA output: " + output)
        return [output]
    except Exception as e:
        logger.error(e)