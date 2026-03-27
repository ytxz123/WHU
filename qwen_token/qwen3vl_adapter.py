from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoProcessor

from .token_formatter import DiscreteMapTokenFormatter


def strip_image_placeholder(text: str) -> str:
    """去掉旧数据里常见的 <image> 占位符。"""
    return str(text).replace("<image>\n", "", 1).replace("<image>", "", 1).strip()


def build_prompt_conversation(
    user_text: str,
    image_path: str,
    formatter: DiscreteMapTokenFormatter,
) -> List[Dict[str, Any]]:
    """构造仅用于推理的 system + user 对话。"""
    return [
        {"role": "system", "content": [{"type": "text", "text": formatter.build_system_prompt()}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": strip_image_placeholder(user_text)},
            ],
        },
    ]


def build_training_conversation(
    user_text: str,
    image_path: str,
    assistant_lines: Sequence[Dict[str, Any]],
    formatter: DiscreteMapTokenFormatter,
) -> List[Dict[str, Any]]:
    """构造包含目标 token 文本的完整对话。"""
    conversation = build_prompt_conversation(
        user_text=user_text,
        image_path=image_path,
        formatter=formatter,
    )
    conversation.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": formatter.lines_to_text(assistant_lines)}],
        }
    )
    return conversation


def render_chat_text(processor: Any, conversation: Sequence[Dict[str, Any]], add_generation_prompt: bool) -> str:
    """把对话渲染成模型需要的聊天模板文本。"""
    return str(
        processor.apply_chat_template(
            list(conversation),
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )
    )


def load_qwen3vl_processor(model_name_or_path: str, formatter: DiscreteMapTokenFormatter) -> AutoProcessor:
    """加载 processor，并注册离散 token。"""
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    formatter.register_tokens_with_processor(processor)
    return processor


def initialize_new_token_embeddings(model: torch.nn.Module, old_vocab_size: int) -> None:
    """用旧词表均值初始化新增 token embedding。"""
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return

    new_vocab_size = int(input_embeddings.weight.shape[0])
    if new_vocab_size <= int(old_vocab_size):
        return

    with torch.no_grad():
        avg_input = input_embeddings.weight[: int(old_vocab_size)].mean(dim=0, keepdim=True)
        input_embeddings.weight[int(old_vocab_size) : new_vocab_size] = avg_input

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.shape[0] == new_vocab_size:
            avg_output = output_embeddings.weight[: int(old_vocab_size)].mean(dim=0, keepdim=True)
            output_embeddings.weight[int(old_vocab_size) : new_vocab_size] = avg_output


def align_model_token_embeddings(model: torch.nn.Module, processor: Any) -> int:
    """让模型词表与 processor.tokenizer 保持一致。"""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose a tokenizer.")

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("Model does not expose input embeddings.")

    old_vocab_size = int(input_embeddings.weight.shape[0])
    new_vocab_size = int(len(tokenizer))
    if new_vocab_size == old_vocab_size:
        return 0

    model.resize_token_embeddings(new_vocab_size)
    initialize_new_token_embeddings(model, old_vocab_size=old_vocab_size)

    final_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    if hasattr(model, "config"):
        model.config.vocab_size = final_vocab_size
    if hasattr(model, "vocab_size"):
        model.vocab_size = final_vocab_size
    return final_vocab_size - old_vocab_size


def decode_generation_output(
    processor: Any,
    generated_sequences: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    formatter: DiscreteMapTokenFormatter,
) -> Dict[str, Any]:
    """把 generate 输出切掉 prompt，并解码成离散 lines。"""
    prompt_length = int(prompt_input_ids.shape[-1])
    generated_ids = generated_sequences[:, prompt_length:]
    decoded_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    token_text = formatter.normalize_generated_text(decoded_text)
    lines = formatter.text_to_lines(token_text)
    return {
        "raw_text": decoded_text,
        "token_text": token_text,
        "lines": lines,
    }


def save_runtime_config(output_dir: str | Path, formatter: DiscreteMapTokenFormatter) -> Path:
    """保存当前离散 token 配置，方便别的框架读取。"""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    config = {
        "image_size": int(formatter.image_size),
        "coord_num_bins": int(formatter.coord_num_bins),
        "token_schema": str(formatter.coordinate_token_style),
        "categories": list(formatter.categories),
        "include_text_prompt_tokens": bool(formatter.include_text_prompt_tokens),
        "num_discrete_tokens": len(formatter.map_tokenizer.itos),
    }
    config_path = output_path / "discrete_token_runtime.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path