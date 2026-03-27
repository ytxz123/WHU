# Qwen3VL-4B 离散 Token 接入说明

这份文档解释一个非常具体的目标：

- 你已经有自己的训练框架和数据集
- 你不想继续沿用原项目训练入口
- 你只想把“离散 token 方案”接到另一套 Qwen3VL-4B 模型里

这个目录就是为这个目标准备的。

## 1. 这套代码解决什么，不解决什么

这个目录负责：

1. 定义离散 token 词表
2. 把结构化 lines 编码成 token 文本
3. 把生成 token 文本解码回 lines
4. 把 token 注册进 Qwen3VL processor.tokenizer
5. 让模型 embedding 跟新增 token 对齐
6. 构造 Qwen3VL 所需的图文对话
7. 保存一份 runtime 配置，方便你自己的训练框架读取

这个目录不负责：

1. 数据集读取
2. 训练 dataloader
3. labels masking
4. Trainer、DeepSpeed、FSDP、LoRA 框架管理
5. checkpoint 保存策略

换句话说，这是一层“离散 token 协议 + Qwen3VL 适配层”，不是完整训练工程。

## 2. 文件结构为什么这样拆

### 2.1 token_formatter.py

文件：

- [qwen3vl_discrete_token_kit/token_formatter.py](qwen3vl_discrete_token_kit/token_formatter.py)

职责只有一个：

- 管离散 token 本身

里面包含：

1. token 词表定义
2. 坐标量化 / 反量化
3. lines -> token text
4. token text -> lines
5. system prompt 构造

这个文件是整个方案的核心，不依赖具体训练框架。

### 2.2 qwen3vl_adapter.py

文件：

- [qwen3vl_discrete_token_kit/qwen3vl_adapter.py](qwen3vl_discrete_token_kit/qwen3vl_adapter.py)

职责只有一个：

- 把离散 token 方案接到 Qwen3VL

里面包含：

1. prompt conversation 构造
2. training conversation 构造
3. chat template 渲染
4. processor 加载和 token 注册
5. 模型 embedding 对齐
6. generate 输出结果解码
7. runtime 配置保存

### 2.3 example_usage.py

文件：

- [qwen3vl_discrete_token_kit/example_usage.py](qwen3vl_discrete_token_kit/example_usage.py)

只提供两类最小示例：

1. 纯离散 token 编解码闭环
2. Qwen3VL prompt 接入示例

它不是训练脚本。

## 3. 你真正要接入的主流程

### 第一步：构造 formatter

```python
formatter = DiscreteMapTokenFormatter(
    image_size=896,
    categories=["road", "lane_line"],
    coord_num_bins=896,
    coordinate_token_style="shared_numbers",
)
```

这一步决定了：

1. 词表里有哪些 token
2. 坐标量化规则是什么
3. 类别 token 有哪些
4. 系统提示词长什么样

### 第二步：加载 Qwen3VL processor，并注册 token

```python
processor = load_qwen3vl_processor(model_path, formatter)
```

内部做的事是：

1. AutoProcessor.from_pretrained
2. formatter.register_tokens_with_processor

这一步不是可选项。

如果你只是把离散 token 写进文本里，但没有真的注册进 tokenizer，这些 token 在模型看来只是普通分词碎片，不是你定义的离散 token 单元。

### 第三步：加载模型并对齐 embedding

```python
model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True)
align_model_token_embeddings(model, processor)
```

这一步会：

1. 读取 tokenizer 当前长度
2. 对比模型旧词表长度
3. 调 resize_token_embeddings
4. 用旧词表均值初始化新增 token embedding

如果你不做这一步，新增 token 不会和模型参数正确对齐。

### 第四步：把 GT lines 编码成 token 文本

```python
token_text = formatter.lines_to_text(lines)
```

这是你训练目标文本的来源。

如果你自己的训练框架要做 supervised fine-tuning，那么 assistant 端目标通常就应该是这串 token 文本。

### 第五步：构造对话

推理用：

```python
conversation = build_prompt_conversation(user_text, image_path, formatter)
```

训练用：

```python
conversation = build_training_conversation(user_text, image_path, assistant_lines, formatter)
```

差别只有一件事：

- training conversation 会多一条 assistant 消息，内容是 GT 离散 token 文本

### 第六步：渲染 chat template

```python
prompt_text = render_chat_text(processor, conversation, add_generation_prompt=True)
```

或者：

```python
full_text = render_chat_text(processor, conversation, add_generation_prompt=False)
```

这里保留成独立函数，是为了让你自己的训练框架可以自己决定：

- 是否单独保留 prompt_text
- 是否构造 prompt/full 两份文本做 masking
- 是否直接走自定义 loss 逻辑

## 4. token_formatter.py 的逻辑怎么理解

### 4.1 词表组成

当前默认词表由 4 部分组成：

1. 基础结构 token
   pad、bos、eos、line、pts、eol
2. 可选文本提示 token
   txt_xy、txt_trace、txt_end
3. 类别 token
   cat_road、cat_lane_line 等
4. 坐标 token
   shared_numbers 或 legacy_xy

### 4.2 坐标 token 风格

支持两种：

1. shared_numbers
   形式是 0 到 N 的共享数字 token，按 x y x y 排列
2. legacy_xy
   形式是 x_i / y_i

如果你没有历史兼容包袱，建议继续用 shared_numbers，词表更小，逻辑也更简单。

### 4.3 一条 line 怎么编码

编码后的结构是：

```text
<bos>
<line> <cat_road> <s_start> <e_end> <pts> <12> <48> <13> <49> <14> <51> <eol>
<eos>
```

如果有多条 line，就重复多个 line 块。

### 4.4 解码为什么需要 normalize

模型 generate 出来的文本可能混入其他字符。

所以推荐流程是：

1. batch_decode 得到 raw_text
2. formatter.normalize_generated_text(raw_text)
3. formatter.text_to_lines(token_text)

这样可以最大程度保证只用你认识的合法 token 解码。

## 5. qwen3vl_adapter.py 的逻辑怎么理解

### 5.1 build_prompt_conversation

输入：

1. user_text
2. image_path
3. formatter

输出：

- system + user 两轮对话

system 内容来自 formatter.build_system_prompt。

### 5.2 build_training_conversation

它在 prompt conversation 基础上，再加一条 assistant 消息。

assistant 文本不是自然语言，而是：

- formatter.lines_to_text(assistant_lines)

### 5.3 load_qwen3vl_processor

它只做两步：

1. 加载 processor
2. 注册离散 token

### 5.4 align_model_token_embeddings

这是接任何训练框架都必须保留的函数。

只要你新增了离散 token，模型 embedding 必须和 tokenizer 对齐。

### 5.5 decode_generation_output

它假设你已经拿到了：

1. generated_sequences
2. prompt_input_ids

函数会自动：

1. 切掉 prompt 段
2. decode 成 raw_text
3. 提取合法 token 文本
4. 解码回 lines

## 6. 你自己的训练框架应该怎么接

这里给一个推荐的职责分界。

### 这套工具包负责

1. token 定义
2. token 编解码
3. Qwen3VL tokenizer 注册
4. model embedding 对齐
5. conversation 构造
6. 生成结果解码

### 你自己的训练框架负责

1. 数据集读取
2. batch 组织
3. prompt/full 双文本编码
4. labels masking
5. loss 计算
6. 优化器和训练策略
7. checkpoint 和 resume

这个边界比较稳，因为离散 token 规则不该被训练框架细节污染。

## 7. 如果你要新增 token，先改哪里

### 场景 A：只加类别

只要：

1. formatter 初始化时传新的 categories
2. 你的数据集 line.category 与之完全一致

通常不需要改 token_formatter.py 结构。

### 场景 B：改坐标 token 风格

只要改 formatter 初始化参数：

1. coordinate_token_style
2. coord_num_bins

但训练和推理必须保持一致。

### 场景 C：新增新的结构 token

比如：

- direction token
- topology token
- confidence token

那就必须改：

1. token_formatter.py 的词表定义
2. encode_lines
3. decode_to_lines
4. build_system_prompt

## 8. 我建议你接入前先做的 3 个验证

### 验证 1：编解码闭环

对一批 GT lines 做：

1. lines_to_text
2. text_to_lines

确认类别、点数、坐标误差都合理。

### 验证 2：processor 注册结果

检查 tokenizer 是否真的包含新 token。

如果没有，后面的训练都没有意义。

### 验证 3：embedding 对齐结果

检查：

1. len(processor.tokenizer)
2. model.get_input_embeddings().weight.shape[0]

两者必须一致。

## 9. 最后一句话总结

你现在要的不是一个新的训练工程，而是一层可移植的离散 token 适配层。

这个目录就是按这个目标设计的：

- 尽量少文件
- 命名直接
- 逻辑拆干净
- 不绑训练框架
- 只保留接 Qwen3VL 必需的那部分实现