# src/lmstudio.py

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "http://localhost:1234/v1"


@dataclass
class VLMResponse:
    """
    Complete result of one VLM inference request.
    """

    model: str
    content: str
    parsed: dict[str, Any] | None
    elapsed_seconds: float

    prompt: str
    image_path: str

    raw_response: Any = None


def get_client(
    base_url: str = DEFAULT_BASE_URL,
) -> OpenAI:
    """
    Create an OpenAI-compatible client for LM Studio.
    """

    return OpenAI(
        base_url=base_url,
        api_key="lm-studio",
    )


def image_to_data_url(
    path: str | Path,
) -> str:
    """
    Encode an image as a base64 data URL.

    PNG, JPEG and WebP are preferred for VLM input.
    """

    path = Path(path)

    mime_type, _ = mimetypes.guess_type(path)

    if mime_type is None:
        raise ValueError(
            f"Unable to determine MIME type: {path}"
        )

    if mime_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        raise ValueError(
            f"Unsupported VLM image type: {mime_type}. "
            "Create a PNG/JPEG/WebP derivative first."
        )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return (
        f"data:{mime_type};base64,{encoded}"
    )


def extract_json(
    content: str,
) -> dict[str, Any]:
    """
    Extract JSON from a model response.

    Handles both raw JSON and:

        ```json
        {...}
        ```
    """

    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model response was not valid JSON.\n\n"
            f"{content}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Expected top-level JSON object."
        )

    return result


def analyze_image(
    image_path: str | Path,
    prompt: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    parse_json: bool = True,
    client: OpenAI | None = None,
) -> VLMResponse:
    """
    Submit one image + prompt to a vision-language model
    running in LM Studio.
    """

    image_path = Path(image_path)

    if client is None:
        client = get_client()

    data_url = image_to_data_url(
        image_path
    )

    start = perf_counter()

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            }
        ],
    )

    elapsed = perf_counter() - start

    content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    parsed = None

    if parse_json:
        parsed = extract_json(content)

    return VLMResponse(
        model=model,
        content=content,
        parsed=parsed,
        elapsed_seconds=elapsed,
        prompt=prompt,
        image_path=str(image_path),
        raw_response=response,
    )


def load_prompt(
    path: str | Path,
) -> str:
    """
    Load a versioned prompt from disk.
    """

    return Path(path).read_text(
        encoding="utf-8"
    )


def save_result(
    response: VLMResponse,
    destination: str | Path,
    *,
    prompt_version: str | None = None,
    task: str | None = None,
) -> Path:
    """
    Save a reproducible experiment result.

    The enormous raw API response is intentionally not serialized.
    """

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "model": response.model,
        "task": task,
        "prompt_version": prompt_version,
        "image": response.image_path,
        "elapsed_seconds": response.elapsed_seconds,
        "prompt": response.prompt,
        "response": response.parsed,
        "raw_content": response.content,
    }

    destination.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return destination