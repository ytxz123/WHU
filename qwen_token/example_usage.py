from __future__ import annotations

from pprint import pprint

from .qwen3vl_adapter import (
    build_prompt_conversation,
    build_training_conversation,
    load_qwen3vl_processor,
    render_chat_text,
    save_runtime_config,
)
from .token_formatter import DiscreteMapTokenFormatter


def demo_roundtrip() -> None:
    # 这段示例只演示离散 token 闭环，不依赖训练框架。
    formatter = DiscreteMapTokenFormatter(
        image_size=896,
        categories=["road", "lane_line"],
        coord_num_bins=896,
        coordinate_token_style="shared_numbers",
    )

    sample_lines = [
        {
            "category": "road",
            "start_type": "start",
            "end_type": "end",
            "points": [[32, 64], [40, 72], [56, 88]],
        },
        {
            "category": "lane_line",
            "start_type": "cut",
            "end_type": "cut",
            "points": [[128, 200], [160, 220]],
        },
    ]

    token_text = formatter.lines_to_text(sample_lines)
    decoded_lines = formatter.text_to_lines(token_text)

    print("=== token text ===")
    print(token_text)
    print("=== decoded lines ===")
    pprint(decoded_lines)


def demo_qwen3vl_prompt(model_path: str) -> None:
    # 这段示例展示怎样把离散 token 接到 Qwen3VL 的 processor。
    formatter = DiscreteMapTokenFormatter(categories=["road"])
    processor = load_qwen3vl_processor(model_path, formatter=formatter)

    prompt_conversation = build_prompt_conversation(
        user_text="请重建这张 patch 内的完整道路拓扑。",
        image_path="/absolute/path/to/image.png",
        formatter=formatter,
    )
    training_conversation = build_training_conversation(
        user_text="请重建这张 patch 内的完整道路拓扑。",
        image_path="/absolute/path/to/image.png",
        assistant_lines=[
            {
                "category": "road",
                "start_type": "start",
                "end_type": "end",
                "points": [[32, 64], [40, 72], [56, 88]],
            }
        ],
        formatter=formatter,
    )

    print("=== prompt text ===")
    print(render_chat_text(processor, prompt_conversation, add_generation_prompt=True))
    print("=== training text ===")
    print(render_chat_text(processor, training_conversation, add_generation_prompt=False))

    config_path = save_runtime_config("./tmp_discrete_runtime", formatter=formatter)
    print(f"runtime config saved to: {config_path}")


if __name__ == "__main__":
    demo_roundtrip()