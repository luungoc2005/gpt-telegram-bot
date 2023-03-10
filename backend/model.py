import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import json
import torch
import logging
import datetime

logger = logging.getLogger(__name__)
handler = logging.FileHandler('model.log')
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Settings
LOAD_SHARDED = False
ENABLE_RANKER = True

# Constants
CACHE_DIR = "G:\\Projects\\Samples\\tmp"
INITIAL_PROMPT = """{{bot_name}} is an assistant chatbot
{{bot_name}} can engage in discussions and answer questions on a wide range of topics

{{bot_name}}: Good {{part_of_day}}
{{chat_history}}"""
# INITIAL_PROMPT = """{{bot_name}} is a large language model. It can talk about anything.

# {{first_name}}: Hello
# {{chat_history}}"""
INITIAL_PROMPT = """A conversation where {{first_name}} interacts with {{bot_name}} 
{{bot_name}} is an AI that's helpful, obedient and honest. It tries to provide diverse responses to keep the conversation engaging.

{{first_name}}: Good {{part_of_day}}
{{bot_name}}: Hello! How can I help you today?
{{chat_history}}"""
RANKER_MODELS = [
    # ('microsoft/DialogRPT-human-vs-rand', .5, 0),
    # ('microsoft/DialogRPT-human-vs-machine', .5, 0),
    ('microsoft/DialogRPT-updown', 1, 1),
    # ('microsoft/DialogRPT-depth', 0.48, 1),
    # ('microsoft/DialogRPT-width', -0.5, 1),
]

MAX_MEMORY = {0: "8GiB", "cpu": "16GiB"}

# Temp variables
_ranker_models = {}


def get_part_of_day():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"


def get_initial_prompt(first_name):
    BOT_NAME = "Sydney"
    prompt = INITIAL_PROMPT \
        .replace("{{first_name}}", first_name) \
        .replace("{{bot_name}}", BOT_NAME) \
        .replace("{{part_of_day}}", get_part_of_day())
    bot_prefix = f"{BOT_NAME}: "
    human_prefix = f"{first_name}: "

    return prompt, bot_prefix, human_prefix

@torch.no_grad()
def rank_candidates(context, candidates):
    global RANKER_MODELS, _ranker_models
    END_TEXT = "<|endoftext|>"
    RANK_DEVICE = "cpu"

    if len(candidates) <= 1 or len(RANKER_MODELS) == 0:
        return 0, torch.zeros((len(candidates),))
    
    responses = [context + END_TEXT + candidate for candidate in candidates]
    num_return_sequences = len(candidates)
    max_idx = 0
    total_results = []

    prior_results = torch.zeros((num_return_sequences,))
    cond_results = torch.zeros((num_return_sequences,))
    has_prior = False
    has_cond = False
    for (ranker, weight, ranker_type) in RANKER_MODELS:

        if ranker not in _ranker_models:
            _ranker_models[ranker] = {
                'model': AutoModelForSequenceClassification.from_pretrained(ranker, cache_dir=CACHE_DIR).to(RANK_DEVICE),
                'tokenizer': AutoTokenizer.from_pretrained(ranker, cache_dir=CACHE_DIR),
            }
        
        ranker_model = _ranker_models[ranker]['model']
        ranker_tokenizer = _ranker_models[ranker]['tokenizer']

        input_ids = ranker_tokenizer(responses, padding=True, return_tensors="pt").to(RANK_DEVICE)

        ranker_results = ranker_model(**input_ids, return_dict=True)
        ranker_results = torch.sigmoid(ranker_results.logits).squeeze()
        if ranker_type == 1:
            prior_results += weight * ranker_results
            has_prior = True
        else:
            cond_results += weight * cond_results
            has_cond = True

    if has_cond and has_prior:
        total_results = prior_results * cond_results
    elif has_cond:
        total_results = cond_results
    else:
        total_results = prior_results

    max_idx = int(torch.argmax(total_results))

    return max_idx, total_results


def process_output(prompt, text_output, stop_words=[]):
    processed_output = text_output[len(prompt):]

    for stop_word in stop_words:
        if stop_word in processed_output:
            processed_output = processed_output[:processed_output.index(stop_word)]
    
    processed_output = processed_output.strip()
    return processed_output


def get_model():
    global INITIAL_PROMPT, LOAD_SHARDED

    # checkpoint = "EleutherAI/gpt-j-6B"
    # checkpoint = "facebook/opt-2.7b"
    checkpoint = "EleutherAI/gpt-neo-2.7B"
    # checkpoint = "KoboldAI/OPT-2.7B-Nerys-v2"
    # checkpoint = "bigscience/bloomz-3b"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    checkpoint_device_map_name = checkpoint.replace("/", "-") + ".json"

    if LOAD_SHARDED:
        config = AutoConfig.from_pretrained(checkpoint)

        if os.path.exists(checkpoint_device_map_name):
            with open(checkpoint_device_map_name) as f:
                device_map = json.load(f)
        else:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config,
                    cache_dir=CACHE_DIR)

            device_map = infer_auto_device_map(model,
                max_memory=MAX_MEMORY)
            print(device_map)
            with open(checkpoint_device_map_name, "w") as f:
                json.dump(device_map, f)

        model = load_checkpoint_and_dispatch(
            model, "sharded-gpt-j-6B", 
            device_map=device_map, 
            no_split_module_classes=["GPTJBlock"],
            offload_folder=CACHE_DIR
        )
    else:
        args = {
            "cache_dir": CACHE_DIR,
        }
        LOAD_IN_8BIT = False
        if torch.cuda.is_available:
            args["torch_dtype"] = torch.float16

            if LOAD_IN_8BIT:
                args["load_in_8bit"] = True
                args["device_map"] = "auto"
                args["max_memory"] = MAX_MEMORY

            else:
                if os.path.exists(checkpoint_device_map_name):
                    with open(checkpoint_device_map_name) as f:
                        device_map = json.load(f)
                else:
                    with init_empty_weights():
                        model = AutoModelForCausalLM.from_pretrained(checkpoint, **args)
                    
                    device_map = infer_auto_device_map(model,
                        max_memory=MAX_MEMORY)

                    print(device_map)
                    with open(checkpoint_device_map_name, "w") as f:
                        json.dump(device_map, f)

                args["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **args)
    model.eval()

    device = next(model.parameters()).device

    return model, tokenizer, device


def build_chat_history(history_items, human_prefix, bot_prefix):
    chat_history = []
    logger.debug([item['message'] for item in history_items])
    for item in history_items:
        message = item['message'].strip()
        if message == "/reset":
            chat_history = []
            continue
        elif message.startswith("/"):
            continue

        if int(item['from']) == 0:
            chat_history.append(human_prefix + message)
        else:
            chat_history.append(bot_prefix + message)
    logger.debug(chat_history)
    return '\n'.join(chat_history)


@torch.no_grad()
def _model_generate(prompt, model, tokenizer, device):
    num_return_sequences=3
    num_beams = num_return_sequences * 2

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)

    output = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=64,
        top_k=50,
        top_p=0.8,
        temperature=0.75,
        repetition_penalty=2.0,
        # diversity_penalty=2.0,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        num_beams=num_beams,
        # num_beam_groups=num_return_sequences,
        do_sample=True,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )
    text_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)

    return text_outputs


def predict(model, tokenizer, device, first_name, history_items, message, generate_func=_model_generate):
    # Prepare input
    initial_prompt, bot_prefix, human_prefix = get_initial_prompt(first_name)
    chat_history = build_chat_history(history_items, human_prefix, bot_prefix)
    prompt = initial_prompt
    rank_context = message

    prompt = initial_prompt \
        .replace("{{chat_history}}", chat_history) \
        .strip() # this prevents a new line if the chat history is empty
    prompt += "\n" + human_prefix + message
    prompt += "\n" + bot_prefix

    logger.debug("Full prompt:")
    logger.debug(prompt)

    text_outputs = generate_func(prompt, model, tokenizer, device)
    candidates = []
    for _, text_output in enumerate(text_outputs):
        processed_output = process_output(prompt, text_output, [human_prefix, '\n'])
        candidates.append(processed_output)
    candidates = list(set(candidates))

    chosen_index, scores = rank_candidates(rank_context, candidates)
    chosen_output = candidates[chosen_index]

    output_lines = [f"{scores[ix]} - {candidate}" for ix, candidate in enumerate(candidates)]
    logger.debug("Candidates:")
    logger.debug("\n".join(output_lines))

    return chosen_output