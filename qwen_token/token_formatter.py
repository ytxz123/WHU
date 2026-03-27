from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


BASE_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<line>",
    "<pts>",
    "<eol>",
]
TEXT_PROMPT_SPECIAL_TOKENS = ["<txt_xy>", "<txt_trace>", "<txt_end>"]
START_TYPES = ["start", "cut"]
END_TYPES = ["end", "cut"]
TOKEN_RE = re.compile(r"<[^<>\s]+>")
DEFAULT_CATEGORIES = ("road",)


def _normalize_categories(categories: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not categories:
        return DEFAULT_CATEGORIES
    values = [str(item).strip() for item in categories if str(item).strip()]
    return tuple(values) if values else DEFAULT_CATEGORIES


class MapSequenceTokenizer:
    """把线段结构编码成离散 token 序列。"""

    def __init__(
        self,
        image_size: int,
        categories: Sequence[str],
        max_seq_len: int,
        coord_num_bins: int = 896,
        coordinate_token_style: str = "shared_numbers",
        include_text_prompt_tokens: bool = False,
    ) -> None:
        self.image_size = int(image_size)
        self.categories = list(categories)
        self.max_seq_len = int(max_seq_len)
        self.coord_num_bins = max(2, int(coord_num_bins))
        self.coordinate_token_style = str(coordinate_token_style).strip().lower()
        self.include_text_prompt_tokens = bool(include_text_prompt_tokens)
        if self.coordinate_token_style not in {"legacy_xy", "shared_numbers"}:
            raise ValueError(f"Unsupported coordinate_token_style: {coordinate_token_style}")

        self.itos = list(BASE_SPECIAL_TOKENS)
        if self.include_text_prompt_tokens:
            self.itos.extend(TEXT_PROMPT_SPECIAL_TOKENS)
        self.itos.extend([f"<cat_{category}>" for category in self.categories])
        self.itos.extend([f"<s_{item}>" for item in START_TYPES])
        self.itos.extend([f"<e_{item}>" for item in END_TYPES])
        if self.coordinate_token_style == "legacy_xy":
            self.itos.extend([f"<x_{idx}>" for idx in range(self.coord_num_bins)])
            self.itos.extend([f"<y_{idx}>" for idx in range(self.coord_num_bins)])
        else:
            self.itos.extend([f"<{idx}>" for idx in range(self.coord_num_bins)])

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def encode_lines(self, lines: Sequence[Dict[str, Any]]) -> List[int]:
        token_ids = [self.bos_id]
        for line in lines:
            category = str(line.get("category", self.categories[0]))
            cat_token = f"<cat_{category}>"
            if cat_token not in self.stoi:
                continue

            points = np.asarray(line.get("points", []), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
                continue

            token_ids.append(self.stoi["<line>"])
            token_ids.append(self.stoi[cat_token])
            token_ids.append(self.stoi.get(f"<s_{line.get('start_type', 'start')}>", self.stoi["<s_start>"]))
            token_ids.append(self.stoi.get(f"<e_{line.get('end_type', 'end')}>", self.stoi["<e_end>"]))
            token_ids.append(self.stoi["<pts>"])
            for x_coord, y_coord in points:
                x_bin = self._quantize_coord(float(x_coord))
                y_bin = self._quantize_coord(float(y_coord))
                token_ids.extend(self._encode_xy_pair(x_bin=x_bin, y_bin=y_bin))
            token_ids.append(self.stoi["<eol>"])

            if len(token_ids) >= self.max_seq_len - 1:
                break

        token_ids = token_ids[: self.max_seq_len - 1]
        token_ids.append(self.eos_id)
        return token_ids

    def decode_to_lines(self, token_ids: Sequence[int]) -> List[Dict[str, np.ndarray]]:
        lines: List[Dict[str, np.ndarray]] = []
        current_category: Optional[str] = None
        current_start_type = "start"
        current_end_type = "end"
        current_points: List[Tuple[float, float]] = []
        reading_points = False
        x_buffer: Optional[float] = None

        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.itos):
                continue
            token = self.itos[int(token_id)]

            if token == "<eos>":
                break
            if token == "<line>":
                self._flush_line(lines, current_category, current_start_type, current_end_type, current_points)
                current_category = None
                current_start_type = "start"
                current_end_type = "end"
                current_points = []
                reading_points = False
                x_buffer = None
                continue
            if token.startswith("<cat_"):
                current_category = token[len("<cat_") : -1]
                continue
            if token.startswith("<s_"):
                current_start_type = token[len("<s_") : -1]
                continue
            if token.startswith("<e_"):
                current_end_type = token[len("<e_") : -1]
                continue
            if token == "<pts>":
                reading_points = True
                continue
            if token == "<eol>":
                self._flush_line(lines, current_category, current_start_type, current_end_type, current_points)
                current_category = None
                current_start_type = "start"
                current_end_type = "end"
                current_points = []
                reading_points = False
                x_buffer = None
                continue
            if not reading_points:
                continue

            coord_value = self._decode_coord_token(token)
            if coord_value is None:
                continue
            if x_buffer is None:
                x_buffer = self._dequantize_coord(coord_value)
                continue
            current_points.append((x_buffer, self._dequantize_coord(coord_value)))
            x_buffer = None

        return lines

    def _flush_line(
        self,
        out_lines: List[Dict[str, np.ndarray]],
        category: Optional[str],
        start_type: str,
        end_type: str,
        points: List[Tuple[float, float]],
    ) -> None:
        if category is None or not points:
            return
        out_lines.append(
            {
                "category": category,
                "start_type": start_type,
                "end_type": end_type,
                "points": np.asarray(points, dtype=np.float32),
            }
        )

    def _encode_xy_pair(self, x_bin: int, y_bin: int) -> List[int]:
        if self.coordinate_token_style == "legacy_xy":
            return [self.stoi[f"<x_{x_bin}>"], self.stoi[f"<y_{y_bin}>"]]
        return [self.stoi[f"<{x_bin}>"], self.stoi[f"<{y_bin}>"]]

    def _decode_coord_token(self, token: str) -> Optional[int]:
        if self.coordinate_token_style == "legacy_xy":
            if token.startswith("<x_") or token.startswith("<y_"):
                return int(np.clip(int(token[3:-1]), 0, self.coord_num_bins - 1))
            return None
        if token.startswith("<") and token.endswith(">"):
            raw = token[1:-1]
            if raw.isdigit():
                return int(np.clip(int(raw), 0, self.coord_num_bins - 1))
        return None

    def _quantize_coord(self, value: float) -> int:
        value = float(np.clip(value, 0.0, float(self.image_size - 1)))
        scale = float(self.coord_num_bins - 1) / float(self.image_size - 1)
        return int(np.clip(round(value * scale), 0, self.coord_num_bins - 1))

    def _dequantize_coord(self, token_index: int) -> float:
        scale = float(self.image_size - 1) / float(self.coord_num_bins - 1)
        return float(int(token_index) * scale)


@lru_cache(maxsize=16)
def _cached_map_tokenizer(
    image_size: int,
    categories: Tuple[str, ...],
    max_seq_len: int,
    coord_num_bins: int,
    coordinate_token_style: str,
    include_text_prompt_tokens: bool,
) -> MapSequenceTokenizer:
    return MapSequenceTokenizer(
        image_size=image_size,
        categories=categories,
        max_seq_len=max_seq_len,
        coord_num_bins=coord_num_bins,
        coordinate_token_style=coordinate_token_style,
        include_text_prompt_tokens=include_text_prompt_tokens,
    )


class DiscreteMapTokenFormatter:
    """提供词表、系统提示、文本编解码三类能力。"""

    def __init__(
        self,
        image_size: int = 896,
        categories: Optional[Sequence[str]] = None,
        max_seq_len: int = 8192,
        coord_num_bins: int = 896,
        coordinate_token_style: str = "shared_numbers",
        include_text_prompt_tokens: bool = False,
    ) -> None:
        categories_tuple = _normalize_categories(categories)
        self.image_size = int(image_size)
        self.categories = categories_tuple
        self.max_seq_len = int(max_seq_len)
        self.coord_num_bins = int(coord_num_bins)
        self.coordinate_token_style = str(coordinate_token_style).strip().lower()
        self.include_text_prompt_tokens = bool(include_text_prompt_tokens)
        self.map_tokenizer = _cached_map_tokenizer(
            image_size=self.image_size,
            categories=self.categories,
            max_seq_len=self.max_seq_len,
            coord_num_bins=self.coord_num_bins,
            coordinate_token_style=self.coordinate_token_style,
            include_text_prompt_tokens=self.include_text_prompt_tokens,
        )

    def build_system_prompt(self) -> str:
        categories_text = ", ".join(self.categories)
        if self.coordinate_token_style == "shared_numbers":
            coord_schema = "<0> <1> ... 形式的共享数字坐标 token，按 x y x y 排列"
            explicit_schema = "<bos> <line> <cat_...> <s_start|cut> <e_end|cut> <pts> <n_x1> <n_y1> ... <eol> ... <eos>"
        else:
            coord_schema = "<x_i> <y_i> 形式的坐标 token"
            explicit_schema = "<bos> <line> <cat_...> <s_start|cut> <e_end|cut> <pts> <x_i> <y_i> ... <eol> ... <eos>"
        return (
            "你是一个卫星图道路重建助手。\n"
            f"请根据当前 patch 图像，预测完整的 {categories_text} 地图。\n"
            "只输出保留离散 token，不要输出 JSON、解释或自然语言。\n"
            "必须严格使用下面的语法：\n"
            f"{explicit_schema}\n"
            f"坐标 token 规则：{coord_schema}。\n"
            "所有坐标都使用 patch 内局部坐标系。"
        )

    def register_tokens_with_processor(self, processor: Any) -> int:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor does not expose a tokenizer.")
        vocab = tokenizer.get_vocab()
        new_tokens = [token for token in self.map_tokenizer.itos if token not in vocab]
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=False)
        return len(new_tokens)

    def extract_known_tokens(self, text: str) -> List[str]:
        tokens = TOKEN_RE.findall(str(text))
        return [token for token in tokens if token in self.map_tokenizer.stoi]

    def normalize_generated_text(self, text: str) -> str:
        return " ".join(self.extract_known_tokens(text))

    def lines_to_text(self, lines: Sequence[Dict[str, Any]]) -> str:
        token_ids = self.map_tokenizer.encode_lines(lines)
        tokens = [
            self.map_tokenizer.itos[int(token_id)]
            for token_id in token_ids
            if int(token_id) != int(self.map_tokenizer.pad_id)
        ]
        return " ".join(tokens)

    def text_to_lines(self, text: str) -> List[Dict[str, Any]]:
        tokens = self.extract_known_tokens(text)
        if not tokens:
            return []

        token_ids = [int(self.map_tokenizer.stoi[token]) for token in tokens]
        if int(self.map_tokenizer.bos_id) not in token_ids:
            token_ids = [int(self.map_tokenizer.bos_id)] + token_ids
        else:
            token_ids = token_ids[token_ids.index(int(self.map_tokenizer.bos_id)) :]
        if int(self.map_tokenizer.eos_id) not in token_ids:
            token_ids.append(int(self.map_tokenizer.eos_id))

        decoded = self.map_tokenizer.decode_to_lines(token_ids)
        out: List[Dict[str, Any]] = []
        for line in decoded:
            points = np.asarray(line.get("points", []), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
                continue
            points = np.rint(points).astype(np.int32)
            out.append(
                {
                    "category": str(line.get("category", self.categories[0])),
                    "start_type": str(line.get("start_type", "start")),
                    "end_type": str(line.get("end_type", "end")),
                    "points": [[int(x), int(y)] for x, y in points.tolist()],
                }
            )
        return out