# Qwen3VL 离散 Token 极简工具包

这个目录只做一件事：

- 把离散 token 方案从原项目的训练入口里拆出来

它不包含训练框架，不依赖 Hugging Face Trainer，也不假设你必须继续使用原来的训练脚本。

适用场景：

- 你已经有成熟训练框架
- 你已经有自己的数据集
- 你只想复用“离散 token 定义 / 注册 / 编解码 / Qwen3VL 接口适配”

## 目录说明

- [qwen3vl_discrete_token_kit/token_formatter.py](qwen3vl_discrete_token_kit/token_formatter.py)
  离散 token 的核心定义、编码、解码、系统提示词
- [qwen3vl_discrete_token_kit/qwen3vl_adapter.py](qwen3vl_discrete_token_kit/qwen3vl_adapter.py)
  Qwen3VL 接口适配，负责 processor 注册 token、模型词表对齐、对话构造、生成结果解码
- [qwen3vl_discrete_token_kit/example_usage.py](qwen3vl_discrete_token_kit/example_usage.py)
  最小使用示例
- [qwen3vl_discrete_token_kit/DISCRETE_TOKEN_QWEN3VL_ZH.md](qwen3vl_discrete_token_kit/DISCRETE_TOKEN_QWEN3VL_ZH.md)
  详细中文说明文档

## 最小接入流程

1. 构造 formatter
2. 加载 Qwen3VL processor，并注册离散 token
3. 加载模型后，对齐 tokenizer 和 embedding 大小
4. 用 formatter 把 GT lines 编码成 token 文本
5. 用 adapter 构造对话或 prompt 文本
6. 训练和推理时都使用同一份 runtime 配置

## 典型伪代码

```python
from transformers import AutoModelForImageTextToText

from qwen3vl_discrete_token_kit import (
    DiscreteMapTokenFormatter,
    align_model_token_embeddings,
    build_prompt_conversation,
    load_qwen3vl_processor,
    render_chat_text,
)

formatter = DiscreteMapTokenFormatter(
    image_size=896,
    categories=["road", "lane_line"],
    coord_num_bins=896,
    coordinate_token_style="shared_numbers",
)

processor = load_qwen3vl_processor("/path/to/Qwen3VL-4B", formatter)
model = AutoModelForImageTextToText.from_pretrained("/path/to/Qwen3VL-4B", trust_remote_code=True)
align_model_token_embeddings(model, processor)

conversation = build_prompt_conversation(
    user_text="请重建这张 patch 内的完整道路拓扑。",
    image_path="/abs/path/image.png",
    formatter=formatter,
)
prompt_text = render_chat_text(processor, conversation, add_generation_prompt=True)
```

## 你需要自己接的部分

这个目录故意不接管下面这些内容：

- 训练框架
- batch sampler
- loss masking
- optimizer / scheduler
- 多卡并行
- checkpoint 管理

这些应该由你自己的训练系统接入。