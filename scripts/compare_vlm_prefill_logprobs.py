"""Compare VLM prompt logprobs for baseline MITO vs mixed prefix+suffix prefill.

Usage:
    # Start the local inference server first.
    uv run inference @ configs/debug/infer.toml \
        --model.name Qwen/Qwen2.5-VL-3B-Instruct \
        --server.port 8000

    # Then run the built-in synthetic multimodal comparison.
    uv run python scripts/compare_vlm_prefill_logprobs.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct

    # Or compare on a custom image/prompt pair.
    uv run python scripts/compare_vlm_prefill_logprobs.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --image-path /path/to/example.png \
        --prompt-text "Describe the image briefly." \
        --assistant-text "A red square."

The script sends two requests to a running local inference server:
1. Standard `/v1/chat/completions` with the full multimodal conversation.
2. Experimental `/v1/chat/completions/tokens` with the multimodal prompt
   rendered first and the assistant continuation supplied as token IDs.
"""

import argparse
import base64
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

from prime_rl.utils.chat_template import common_prefix_len, render_messages

COLOR_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

DEFAULT_SYSTEM_PROMPT = """You will be shown colored squares across multiple turns. Each color maps to a letter:

Red=A, Green=B, Blue=C, Yellow=D, Purple=E, Cyan=F, Orange=G, White=H, Black=I

Example: Turn 1 shows Red, Blue. Turn 2 shows Green, Yellow. The full codeword is "ACBD" (all 4 letters in order).

After each turn, output your accumulated codeword so far. Output ONLY the letters with NO spaces."""

# Prefix the synthetic assistant text with Qwen's assistant boundary token so
# this test does not depend on newline-sensitive chat-template retokenization at
# the prompt/response seam.
ASSISTANT_RESPONSE = """<|im_start|>
<think>
Let me think about this step by step. 

## Step 1: Identify the colors in each turn and map them to their corresponding letters.
- Turn 1: Red, Blue
  - Red corresponds to A
  - Blue corresponds to C
- Turn 2: Green, Yellow
  - Green corresponds to B
  - Yellow corresponds to D

## Step 2: Combine the letters in the order they were shown across the turns.
- From Turn 1, we have A and C, so we start with "AC".
- From Turn 2, we have B and D, which we add to the end of the codeword, resulting in "ACBD".

## Step 3: Output the final codeword and verify that it matches the expected format.
- The final codeword is "ACBD", which consists of the letters corresponding to the colors shown in order, with no spaces.

## Step 4: Use the memory tool to verify the final answer.
- I will check the memory to confirm that the codeword I derived matches the expected codeword for the given colors.
<memory>
Turn 1: Red, Blue -> A, C
Turn 2: Green, Yellow -> B, D
</memory>

## Step 5: Read the memory and confirm the final answer.
- After reviewing the memory, I confirm that the codeword derived from the colors shown in the turns is indeed "ACBD".

## Final Answer: "ACBD"
</think>

The full codeword is "ACBD".
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prompt logprobs between the baseline multimodal MITO path and an experimental mixed "
            "messages+suffix-tokens prefill path."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for the inference server.")
    parser.add_argument("--model", required=True, help="Model name used by both the local tokenizer and the server.")
    parser.add_argument(
        "--preset",
        choices=["color-codeword"],
        default="color-codeword",
        help="Built-in multimodal test case to use when no explicit image/prompt arguments are provided.",
    )
    parser.add_argument("--image-path", help="Optional local image path for a custom multimodal prompt.")
    parser.add_argument("--prompt-text", help="Optional user text prompt for a custom multimodal prompt.")
    parser.add_argument("--assistant-text", help="Assistant response text to score.")
    parser.add_argument("--system-prompt", help="Optional system prompt for a custom multimodal prompt.")
    parser.add_argument(
        "--prompt-logprobs",
        type=int,
        default=20,
        help="Top-k prompt logprobs to request from the server.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed absolute difference between matching token logprobs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the local processor/tokenizer.",
    )
    return parser.parse_args()


def image_path_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def create_color_image(color: str, size: int = 100):
    from PIL import Image

    return Image.new("RGB", (size, size), COLOR_RGB[color])


def pil_image_to_data_url(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def color_codeword_example() -> tuple[list, list, str]:
    images = [create_color_image("red"), create_color_image("blue")]
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": "Here are 2 squares."},
            ],
        },
    ]
    return messages, images, ASSISTANT_RESPONSE


def to_request_messages(local_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    request_messages: list[dict[str, Any]] = []
    for message in local_messages:
        content = message.get("content")
        if isinstance(content, str):
            request_messages.append({"role": message["role"], "content": content})
            continue

        if all(item.get("type") == "text" for item in content):
            text_content = "".join(item["text"] for item in content)
            request_messages.append({"role": message["role"], "content": text_content})
            continue

        request_content = []
        for item in content:
            if item.get("type") == "image":
                request_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_url(item["image"])}})
            elif item.get("type") == "text":
                request_content.append({"type": "text", "text": item["text"]})
        request_messages.append({"role": message["role"], "content": request_content})
    return request_messages


def build_custom_messages(
    *,
    image_path: Path,
    prompt_text: str,
    assistant_text: str,
    system_prompt: str | None,
) -> tuple[list[dict[str, Any]], str]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    prompt_messages = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    )
    return prompt_messages, assistant_text


def build_local_messages(prompt_messages: list[dict[str, Any]], assistant_text: str | None = None) -> list[dict[str, Any]]:
    if assistant_text is None:
        return prompt_messages

    return prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]


def to_processor_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processor_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            processor_messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": content}],
                }
            )
            continue
        processor_messages.append(message)
    return processor_messages


def load_chat_renderer(model_name: str, trust_remote_code: bool):
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        processor = None
    return tokenizer, processor


def render_ids(tokenizer, processor, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    if processor is not None:
        messages = to_processor_messages(messages)
    return render_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        processor=processor,
    )


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        if body:
            raise requests.HTTPError(f"{exc}\nResponse body: {body}", response=response) from exc
        raise
    return response.json()


def get_token_logprob(entry: dict[str, Any] | None, token_id: int) -> float:
    if entry is None:
        raise ValueError(f"Missing prompt logprob entry for token {token_id}.")
    token_entry = entry.get(str(token_id))
    if token_entry is None:
        token_entry = entry.get(token_id)
    if token_entry is None:
        raise ValueError(f"Prompt logprob entry does not include token {token_id}.")
    return float(token_entry["logprob"])


def token_repr(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(token_id)
    if token is not None:
        return repr(token)
    return repr(tokenizer.decode([token_id], skip_special_tokens=False))


def print_token_window(name: str, tokenizer, token_ids: list[int], center: int, radius: int = 6) -> None:
    start = max(0, center - radius)
    end = min(len(token_ids), center + radius)
    print(f"{name}_window start={start} end={end} center={center}")
    for index in range(start, end):
        print(f"{name}[{index}] id={token_ids[index]} token={token_repr(tokenizer, token_ids[index])}")
    print(f"{name}_decoded={tokenizer.decode(token_ids[start:end], skip_special_tokens=False)!r}")


def print_alignment_debug(
    *,
    tokenizer,
    prompt_only_ids: list[int],
    full_ids: list[int],
    baseline_prompt_ids: list[int],
    mixed_prompt_ids: list[int],
    suffix_tokens: list[int],
    split_idx: int,
) -> None:
    local_prefix = common_prefix_len(prompt_only_ids, full_ids)
    server_prefix = common_prefix_len(mixed_prompt_ids, baseline_prompt_ids)
    local_to_server_prompt_prefix = common_prefix_len(prompt_only_ids, mixed_prompt_ids)
    local_to_server_full_prefix = common_prefix_len(full_ids, baseline_prompt_ids)
    mixed_suffix_start = len(mixed_prompt_ids) - len(suffix_tokens)

    print("debug_alignment")
    print(f"local_prompt_only_len={len(prompt_only_ids)}")
    print(f"local_full_len={len(full_ids)}")
    print(f"server_mixed_prompt_len={len(mixed_prompt_ids)}")
    print(f"server_baseline_prompt_len={len(baseline_prompt_ids)}")
    print(f"local_prefix_len={local_prefix}")
    print(f"server_prefix_len={server_prefix}")
    print(f"local_to_server_prompt_prefix_len={local_to_server_prompt_prefix}")
    print(f"local_to_server_full_prefix_len={local_to_server_full_prefix}")
    print(f"split_idx={split_idx}")
    print(f"server_mixed_suffix_start={mixed_suffix_start}")
    print(f"suffix_len={len(suffix_tokens)}")

    print_token_window("local_prompt_only", tokenizer, prompt_only_ids, local_prefix)
    print_token_window("local_full", tokenizer, full_ids, local_prefix)
    print_token_window("server_mixed_prompt", tokenizer, mixed_prompt_ids, mixed_suffix_start)
    print_token_window("server_baseline_prompt", tokenizer, baseline_prompt_ids, split_idx)


def compare_suffix_logprobs(
    baseline_prompt_logprobs: list[dict[str, Any] | None],
    mixed_prompt_logprobs: list[dict[str, Any] | None],
    suffix_tokens: list[int],
    *,
    baseline_suffix_start: int,
    mixed_suffix_start: int,
    tolerance: float,
) -> tuple[float, int | None]:
    max_abs_diff = 0.0
    mismatch_index: int | None = None

    for index, token_id in enumerate(suffix_tokens):
        baseline_logprob = get_token_logprob(baseline_prompt_logprobs[baseline_suffix_start + index], token_id)
        mixed_logprob = get_token_logprob(mixed_prompt_logprobs[mixed_suffix_start + index], token_id)
        abs_diff = abs(baseline_logprob - mixed_logprob)
        max_abs_diff = max(max_abs_diff, abs_diff)
        if mismatch_index is None and abs_diff > tolerance:
            mismatch_index = index
        print(index, token_id, abs_diff, baseline_logprob, mixed_logprob)
    return max_abs_diff, mismatch_index


def main() -> None:
    args = parse_args()
    if args.image_path:
        if args.prompt_text is None or args.assistant_text is None:
            raise ValueError("--prompt-text and --assistant-text are required when --image-path is provided.")
        image_path = Path(args.image_path).expanduser().resolve()
        prompt_only_local_messages, assistant_text = build_custom_messages(
            image_path=image_path,
            prompt_text=args.prompt_text,
            assistant_text=args.assistant_text,
            system_prompt=args.system_prompt,
        )
        image_label = str(image_path)
    else:
        prompt_only_local_messages, _, assistant_text = color_codeword_example()
        image_label = "color-codeword:red-blue"

    full_local_messages = build_local_messages(prompt_only_local_messages, assistant_text)
    prompt_only_request_messages = to_request_messages(prompt_only_local_messages)
    full_request_messages = to_request_messages(full_local_messages)
    tokenizer, processor = load_chat_renderer(args.model, args.trust_remote_code)

    prompt_only_ids = render_ids(
        tokenizer,
        processor,
        prompt_only_local_messages,
        add_generation_prompt=True,
    )
    full_ids = render_ids(
        tokenizer,
        processor,
        full_local_messages,
        add_generation_prompt=False,
    )
    split_idx = common_prefix_len(prompt_only_ids, full_ids)
    suffix_tokens = full_ids[split_idx:]

    baseline_payload = {
        "model": args.model,
        "messages": full_request_messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "prompt_logprobs": args.prompt_logprobs,
        "return_token_ids": True,
        "skip_special_tokens": False,
    }
    mixed_payload = {
        "model": args.model,
        "messages": prompt_only_request_messages,
        "tokens": suffix_tokens,
        "use_messages": True,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "prompt_logprobs": args.prompt_logprobs,
        "return_token_ids": True,
        "skip_special_tokens": False,
    }

    baseline = post_json(f"{args.base_url.rstrip('/')}/v1/chat/completions", baseline_payload, args.timeout)
    mixed = post_json(f"{args.base_url.rstrip('/')}/v1/chat/completions/tokens", mixed_payload, args.timeout)

    baseline_prompt_logprobs = baseline["prompt_logprobs"]
    mixed_prompt_logprobs = mixed["prompt_logprobs"]
    baseline_prompt_ids = baseline["prompt_token_ids"]
    mixed_prompt_ids = mixed["prompt_token_ids"]

    print_alignment_debug(
        tokenizer=tokenizer,
        prompt_only_ids=prompt_only_ids,
        full_ids=full_ids,
        baseline_prompt_ids=baseline_prompt_ids,
        mixed_prompt_ids=mixed_prompt_ids,
        suffix_tokens=suffix_tokens,
        split_idx=split_idx,
    )

    max_abs_diff, mismatch_index = compare_suffix_logprobs(
        baseline_prompt_logprobs,
        mixed_prompt_logprobs,
        suffix_tokens,
        baseline_suffix_start=split_idx,
        mixed_suffix_start=len(prompt_only_ids),
        tolerance=args.tolerance,
    )

    print(f"example={image_label}")
    print(f"model={args.model}")
    print(f"prompt_only_len={len(prompt_only_ids)}")
    print(f"full_len={len(full_ids)}")
    print(f"common_prefix_len={split_idx}")
    print(f"suffix_len={len(suffix_tokens)}")
    print(f"baseline_suffix_start={split_idx}")
    print(f"mixed_suffix_start={len(prompt_only_ids)}")
    print(f"max_abs_diff={max_abs_diff:.12g}")

    if mismatch_index is None:
        print("result=PASS")
        return

    token_id = suffix_tokens[mismatch_index]
    baseline_logprob = get_token_logprob(baseline_prompt_logprobs[split_idx + mismatch_index], token_id)
    mixed_logprob = get_token_logprob(mixed_prompt_logprobs[len(prompt_only_ids) + mismatch_index], token_id)
    print("result=FAIL")
    print(f"first_mismatch_index={mismatch_index}")
    print(f"first_mismatch_token_id={token_id}")
    print(f"baseline_logprob={baseline_logprob:.12g}")
    print(f"mixed_logprob={mixed_logprob:.12g}")


if __name__ == "__main__":
    main()
