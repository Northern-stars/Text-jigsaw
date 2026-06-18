import argparse
import json
import os
import random
import re
from pathlib import Path

import cv2
import enchant
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


ENGLISH_DICT = enchant.Dict(
    "en_US"
)


def _normalize_word_for_check(
    word: str
) -> str:

    match = re.search(
        r"[A-Za-z][A-Za-z'-]*",
        word
    )

    return match.group(0) if match else ""


def _is_english_word(
    word: str
) -> bool:

    normalized_word = _normalize_word_for_check(
        word
    )

    return bool(
        normalized_word
    ) and ENGLISH_DICT.check(
        normalized_word
    )


def _replace_single_newline(
    prev_line: str,
    next_line: str
) -> str:

    prev_words = prev_line.split()
    next_words = next_line.split()

    if not prev_words or not next_words:
        return ""

    prev_word = prev_words[-1]
    next_word = next_words[0]

    if (
        not _is_english_word(
            prev_word
        )
        or
        not _is_english_word(
            next_word
        )
    ):
        return ""

    return " "


def read_txt_content(
    txt_path: str
) -> str:
    """
    读取txt文件内容，并根据换行上下文处理换行符。
    """

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    parts = re.split(
        r"(\n+)",
        text
    )

    output = []

    for i, part in enumerate(
        parts
    ):

        if not part.startswith(
            "\n"
        ):
            output.append(
                part
            )
            continue

        if len(
            part
        ) >= 2:
            output.append(
                "\n"
            )
            continue

        prev_line = parts[i - 1] if i > 0 else ""
        next_line = parts[i + 1] if i + 1 < len(
            parts
        ) else ""

        output.append(
            _replace_single_newline(
                prev_line,
                next_line
            )
        )

    return "".join(
        output
    )


def split_text_by_max_length(
    text: str,
    max_length: int
) -> list[str]:
    """
    按字符数上限将字符串切分为多个片段。
    """

    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than 0"
        )

    return [
        text[i:i + max_length]
        for i in range(
            0,
            len(text),
            max_length
        )
    ]


def _draw_erosion_edge(
    draw: ImageDraw.ImageDraw,
    piece_w: int,
    piece_h: int,
    erosion_width: int,
    side: str,
):

    if erosion_width <= 0:
        return

    if side in (
        "top",
        "bottom"
    ):
        length = piece_w
    else:
        length = piece_h

    min_step = max(
        4,
        erosion_width // 2
    )

    max_step = max(
        min_step + 1,
        erosion_width * 2
    )

    edge_points = []

    pos = 0

    while pos < length:

        depth = random.randint(
            0,
            erosion_width
        )

        if side == "top":
            edge_points.append(
                (pos, depth)
            )
        elif side == "bottom":
            edge_points.append(
                (pos, piece_h - 1 - depth)
            )
        elif side == "left":
            edge_points.append(
                (depth, pos)
            )
        else:
            edge_points.append(
                (piece_w - 1 - depth, pos)
            )

        pos += random.randint(
            min_step,
            max_step
        )

    if side == "top":
        edge_points.append(
            (piece_w - 1, random.randint(0, erosion_width))
        )
        polygon = [
            (0, 0),
            (piece_w - 1, 0),
            *reversed(edge_points),
        ]
    elif side == "bottom":
        edge_points.append(
            (piece_w - 1, piece_h - 1 - random.randint(0, erosion_width))
        )
        polygon = [
            (0, piece_h - 1),
            (piece_w - 1, piece_h - 1),
            *reversed(edge_points),
        ]
    elif side == "left":
        edge_points.append(
            (random.randint(0, erosion_width), piece_h - 1)
        )
        polygon = [
            (0, 0),
            (0, piece_h - 1),
            *reversed(edge_points),
        ]
    else:
        edge_points.append(
            (piece_w - 1 - random.randint(0, erosion_width), piece_h - 1)
        )
        polygon = [
            (piece_w - 1, 0),
            (piece_w - 1, piece_h - 1),
            *reversed(edge_points),
        ]

    draw.polygon(
        polygon,
        fill="black"
    )


def _draw_erosion(
    draw: ImageDraw.ImageDraw,
    piece_w: int,
    piece_h: int,
    erosion_width: int,
):

    for side in (
        "top",
        "right",
        "bottom",
        "left"
    ):
        _draw_erosion_edge(
            draw,
            piece_w,
            piece_h,
            erosion_width,
            side
        )


def _fade(
    value
):

    return (
        6 * value ** 5
        - 15 * value ** 4
        + 10 * value ** 3
    )


def _lerp(
    start,
    end,
    weight
):

    return start + weight * (
        end - start
    )


def _perlin_noise(
    width: int,
    height: int,
    cell_size: int = 64
):

    cells_x = max(
        1,
        int(np.ceil(width / cell_size))
    )

    cells_y = max(
        1,
        int(np.ceil(height / cell_size))
    )

    x = np.linspace(
        0,
        cells_x,
        width,
        endpoint=False
    )

    y = np.linspace(
        0,
        cells_y,
        height,
        endpoint=False
    )

    xi = x.astype(int)
    yi = y.astype(int)

    xf = x - xi
    yf = y - yi

    xi_grid, yi_grid = np.meshgrid(
        xi,
        yi
    )

    xf_grid, yf_grid = np.meshgrid(
        xf,
        yf
    )

    angles = np.random.random(
        (cells_y + 1, cells_x + 1)
    ) * 2 * np.pi

    gradients = np.dstack(
        (
            np.cos(angles),
            np.sin(angles)
        )
    )

    g00 = gradients[
        yi_grid,
        xi_grid
    ]

    g10 = gradients[
        yi_grid,
        xi_grid + 1
    ]

    g01 = gradients[
        yi_grid + 1,
        xi_grid
    ]

    g11 = gradients[
        yi_grid + 1,
        xi_grid + 1
    ]

    n00 = (
        g00[..., 0] * xf_grid
        + g00[..., 1] * yf_grid
    )

    n10 = (
        g10[..., 0] * (xf_grid - 1)
        + g10[..., 1] * yf_grid
    )

    n01 = (
        g01[..., 0] * xf_grid
        + g01[..., 1] * (yf_grid - 1)
    )

    n11 = (
        g11[..., 0] * (xf_grid - 1)
        + g11[..., 1] * (yf_grid - 1)
    )

    u = _fade(
        xf_grid
    )

    v = _fade(
        yf_grid
    )

    nx0 = _lerp(
        n00,
        n10,
        u
    )

    nx1 = _lerp(
        n01,
        n11,
        u
    )

    noise = _lerp(
        nx0,
        nx1,
        v
    )

    noise_min = noise.min()
    noise_max = noise.max()

    if noise_max == noise_min:
        return np.zeros(
            (height, width),
            dtype=np.float32
        )

    return (
        (noise - noise_min)
        / (noise_max - noise_min)
    ).astype(
        np.float32
    )


def _create_paper_image(
    width: int,
    height: int,
    paper_yellowing: bool
):

    if not paper_yellowing:
        return Image.new(
            "RGB",
            (width, height),
            "white"
        )

    paper_brightness = random.randint(
        220,
        245
    )

    base_color = np.array(
        [
            paper_brightness,
            paper_brightness - random.randint(8, 18),
            paper_brightness - random.randint(25, 45)
        ],
        dtype=np.float32
    )

    large_noise = _perlin_noise(
        width,
        height,
        cell_size=max(
            32,
            min(width, height) // 2
        )
    )

    small_noise = np.random.normal(
        0,
        3,
        (height, width)
    )

    y, x = np.indices(
        (height, width)
    )

    center_x = (
        width - 1
    ) / 2

    center_y = (
        height - 1
    ) / 2

    radius = np.sqrt(
        (
            (x - center_x)
            / max(center_x, 1)
        ) ** 2
        + (
            (y - center_y)
            / max(center_y, 1)
        ) ** 2
    )

    edge_strength = np.clip(
        radius,
        0,
        1
    ) ** 1.8

    vertical_gradient = (
        y / max(height, 1)
    )

    yellow_strength = (
        large_noise * 10
        + edge_strength * 20
        + vertical_gradient * 4
        + small_noise
    )

    yellow_tint = np.array(
        [1.0, 0.5, -0.3],
        dtype=np.float32
    )

    paper = (
        base_color
        + yellow_strength[..., None] * yellow_tint
    )

    scan_noise = np.random.normal(
        0,
        2,
        (height, width, 1)
    )

    paper += scan_noise

    paper = np.clip(
        paper,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        paper,
        "RGB"
    )


def _get_text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont
) -> int:

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    return bbox[2] - bbox[0]


def _draw_adjusted_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: str,
    left: int,
    usable_w: int
):

    words = line.split()

    if len(words) <= 1:

        line_w = _get_text_width(
            draw,
            line,
            font
        )

        x = left + max(
            0,
            (usable_w - line_w) / 2
        )

        draw.text(
            (x, y),
            line,
            fill=fill,
            font=font
        )

        return

    word_widths = [
        _get_text_width(
            draw,
            word,
            font
        )
        for word in words
    ]

    words_w = sum(
        word_widths
    )

    line_w = _get_text_width(
        draw,
        line,
        font
    )

    if words_w >= usable_w:

        x = left + max(
            0,
            (usable_w - line_w) / 2
        )

        draw.text(
            (x, y),
            line,
            fill=fill,
            font=font
        )

        return

    gap = (
        usable_w - words_w
    ) / (
        len(words) + 1
    )

    x = left + gap

    for word, word_w in zip(
        words,
        word_widths
    ):

        draw.text(
            (x, y),
            word,
            fill=fill,
            font=font
        )

        x += word_w + gap


def generate_text_puzzle(
    page_text: str,
    output_path: str,
    font_path: str,
    page_size=(2480, 3508),
    font_size=32,
    margin=120,
    grid_size=3,
    erosion=False,
    erosion_width=0,
    paper_yellowing=True,
):
    """
    Generate page-level text puzzle.

    Layout strategy:
        Fill page row by row.

        line0:
            piece0 -> piece1 -> piece2

        line1:
            piece0 -> piece1 -> piece2

        ...

    Then extract each piece vertically.
    """

    page_w, page_h = page_size

    piece_w = page_w // grid_size
    piece_h = page_h // grid_size

    edge_margin = margin

    if erosion:

        erosion_width = max(
            0,
            int(erosion_width)
        )

        edge_margin += erosion_width

    else:

        erosion_width = 0

    font = ImageFont.truetype(
        font_path,
        font_size
    )

    dummy_img = Image.new(
        "RGB",
        (100, 100)
    )

    dummy_draw = ImageDraw.Draw(
        dummy_img
    )

    line_height = int(
        font_size * 1.2
    )

    usable_w = (
        piece_w - 2 * edge_margin
    )

    usable_h = (
        piece_h - 2 * edge_margin
    )

    if (
        usable_w <= 0
        or
        usable_h <= 0
    ):
        raise ValueError(
            "margin and erosion_width leave no usable area for text"
        )

    max_piece_lines = max(
        1,
        usable_h // line_height
    )

    # ==================================================
    # tokenize
    # ==================================================

    tokens = re.findall(
        r'\n|[A-Za-z0-9]+|[^\w\s]+| +',
        page_text
    )

    # ==================================================
    # page grid
    # ==================================================

    total_page_rows = (
        max_piece_lines * grid_size
    )

    page_grid = [
        [""] * grid_size
        for _ in range(total_page_rows)
    ]

    row_idx = 0
    col_idx = 0

    current_text = ""

    def flush_segment(
        advance_col: bool = True,
        keep_newline: bool = False
    ):

        nonlocal current_text
        nonlocal row_idx
        nonlocal col_idx

        page_grid[row_idx][col_idx] = (
            current_text.strip()
            + ("\n" if keep_newline else "")
        )

        current_text = ""

        if advance_col:

            col_idx += 1

            if col_idx >= grid_size:

                col_idx = 0
                row_idx += 1

        else:

            row_idx += 1
            col_idx = 0

    # ==================================================
    # fill page
    # ==================================================

    for token in tokens:

        if row_idx >= total_page_rows:
            break

        if token == "\n":

            flush_segment(
                advance_col=False,
                keep_newline=True
            )

            if row_idx >= total_page_rows:
                break

            continue

        candidate = (
            current_text + token
        )

        bbox = dummy_draw.textbbox(
            (0, 0),
            candidate,
            font=font
        )

        width = (
            bbox[2] - bbox[0]
        )

        if width <= usable_w:

            current_text = candidate

        else:

            flush_segment()

            if row_idx >= total_page_rows:
                break

            current_text = (
                token.lstrip()
            )

    if (
        row_idx < total_page_rows
        and current_text.strip()
    ):
        page_grid[row_idx][col_idx] = (
            current_text.strip()
        )

    # ==================================================
    # build piece texts
    # ==================================================

    labels = {
        "grid_size": grid_size,
        "font_size": font_size,
        "erosion": erosion,
        "erosion_width": erosion_width,
        "paper_yellowing": paper_yellowing,
        "paper_effect": (
            "random_base_perlin_edge_scan_noise"
            if paper_yellowing
            else "white"
        ),
        "pieces": []
    }

    paper_img = _create_paper_image(
        page_w,
        page_h,
        paper_yellowing
    )

    for piece_id in range(
        grid_size * grid_size
    ):

        piece_row = (
            piece_id // grid_size
        )

        piece_col = (
            piece_id % grid_size
        )

        render_lines = []
        label_lines = []

        start_row = (
            piece_row * max_piece_lines
        )

        end_row = (
            (piece_row + 1) * max_piece_lines
        )

        for r in range(
            start_row,
            end_row
        ):

            if r >= len(page_grid):
                break

            text = page_grid[r][piece_col]

            render_lines.append(
                text
            )

            if (
                text.strip()
                or
                text == "\n"
            ):
                label_lines.append(
                    text
                )

        # render piece

        img = paper_img.crop(
            (
                piece_col * piece_w,
                piece_row * piece_h,
                (piece_col + 1) * piece_w,
                (piece_row + 1) * piece_h
            )
        )

        draw = ImageDraw.Draw(
            img
        )

        y = edge_margin

        for line in render_lines:

            render_line = line.rstrip(
                "\n"
            )

            if render_line:

                _draw_adjusted_line(
                    draw,
                    render_line,
                    y,
                    font,
                    "black",
                    edge_margin,
                    usable_w
                )

            y += line_height

        if erosion:

            _draw_erosion(
                draw,
                piece_w,
                piece_h,
                erosion_width
            )

        img_path = (
            f"{output_path}_{piece_id}.png"
        )

        img.save(
            img_path
        )

        piece_text ="<SEG>"+ "".join(
            f"{line}<SEG>"
            for line in label_lines
        )

        labels["pieces"].append(
            {
                "piece_id": piece_id,
                "row": piece_row,
                "col": piece_col,
                "text": piece_text,
                "segments": label_lines,
                "image": os.path.basename(
                    img_path
                )
            }
        )

    with open(
        output_path + ".json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            labels,
            f,
            ensure_ascii=False,
            indent=2
        )

    return labels


def visualize_puzzle(
    puzzle_path: str,
    window_name: str = "Puzzle",
    wait_key: int = 0
):
    """
    可视化generate_text_puzzle生成结果

    Parameters
    ----------
    puzzle_path :
        不带.json后缀

    Example
    -------
    visualize_puzzle(
        "./dataset/page_0001"
    )
    """

    with open(
        puzzle_path + ".json",
        "r",
        encoding="utf-8"
    ) as f:

        labels = json.load(f)

    grid_size = labels["grid_size"]

    pieces = sorted(
        labels["pieces"],
        key=lambda x: x["piece_id"]
    )

    if len(pieces) == 0:
        raise ValueError(
            "No pieces found."
        )

    puzzle_dir = os.path.dirname(
        puzzle_path
    )

    first_img_path = os.path.join(
        puzzle_dir,
        pieces[0]["image"]
    )

    first_img = cv2.imread(
        first_img_path
    )

    if first_img is None:
        raise FileNotFoundError(
            first_img_path
        )

    piece_h, piece_w = first_img.shape[:2]

    canvas = np.full(
        (
            piece_h * grid_size,
            piece_w * grid_size,
            3
        ),
        255,
        dtype=np.uint8
    )

    for piece in pieces:

        piece_id = piece["piece_id"]

        row = piece_id // grid_size
        col = piece_id % grid_size

        image_path = os.path.join(
            puzzle_dir,
            piece["image"]
        )

        img = cv2.imread(
            image_path
        )

        if img is None:
            raise FileNotFoundError(
                image_path
            )

        if (
            img.shape[0] != piece_h
            or
            img.shape[1] != piece_w
        ):
            img = cv2.resize(
                img,
                (piece_w, piece_h)
            )

        y1 = row * piece_h
        y2 = y1 + piece_h

        x1 = col * piece_w
        x2 = x1 + piece_w

        canvas[
            y1:y2,
            x1:x2
        ] = img

    cv2.imshow(
        window_name,
        canvas
    )

    cv2.waitKey(
        wait_key
    )

    return canvas


def read_label_text(
    json_path: str
) -> str:

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        labels = json.load(f)

    pieces = sorted(
        labels["pieces"],
        key=lambda x: x["piece_id"]
    )

    output = []

    for piece in pieces:

        output.append(
            f"========== Piece {piece['piece_id']} =========="
        )

        output.append(
            piece["text"]
        )

        output.append("")

    return "\n".join(output)


def _iter_txt_paths(txt_dir: str) -> list[str]:
    txt_dir = os.path.abspath(txt_dir)
    file_paths = []

    for root, _, files in os.walk(txt_dir):
        for file_name in files:
            if file_name.lower().endswith(".txt"):
                file_paths.append(os.path.join(root, file_name))

    return file_paths


def _iter_font_paths(font_dir: str) -> list[str]:
    path = Path(font_dir)
    if not path.exists():
        return []

    font_paths = [
        str(p)
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".ttf", ".otf", ".ttc"}
    ]
    return sorted(font_paths)


def _parse_image_size(image_size) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)

    if isinstance(image_size, tuple):
        return image_size

    text = str(image_size).lower().replace("x", ",").replace(" ", "")
    parts = [p for p in text.split(",") if p]

    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)

    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))

    raise ValueError("image_size must be an int or width,height")


def generate_dataset_from_dirs(
    txt_dir: str,
    font_dir: str,
    segment_chars: int,
    sample_num: int,
    output_dir: str,
    image_size=(576, 576),
    grid_size=3,
    margin=1,
    erosion_width: int | None = None,
):
    txt_paths = _iter_txt_paths(txt_dir)
    if not txt_paths:
        raise ValueError(f"No .txt files found under {txt_dir}")

    font_paths = _iter_font_paths(font_dir)
    if not font_paths:
        raise ValueError(f"No font files found under {font_dir}")

    if segment_chars <= 0:
        raise ValueError("segment_chars must be greater than 0")

    if sample_num <= 0:
        raise ValueError("sample_num must be greater than 0")

    page_size = _parse_image_size(image_size)
    os.makedirs(output_dir, exist_ok=True)

    font_size_labels = {
        "large": max(12, min(page_size) // 20),
        "medium": max(10, min(page_size) // 24),
        "small": max(8, min(page_size) // 30),
    }

    if erosion_width is None:
        erosion_width = max(1, min(page_size) // 120)

    file_cache: dict[str, str] = {}

    def _load_text(path: str) -> str:
        if path not in file_cache:
            file_cache[path] = read_txt_content(path)
        return file_cache[path]

    def _pick_unique_segment(seen_segments: set[str]) -> str:
        max_attempts = 200

        for attempt in range(max_attempts):
            txt_path = random.choice(txt_paths)
            text = _load_text(txt_path)
            if not text:
                continue

            if len(text) <= segment_chars:
                segment = text.strip()
            else:
                start = random.randint(0, len(text) - segment_chars)
                segment = text[start:start + segment_chars].strip()

            if not segment:
                continue

            if segment in seen_segments:
                continue

            return segment

        raise RuntimeError(
            "Unable to select a unique text segment after "
            f"{max_attempts} attempts. Try increasing txt data or lowering sample_num."
        )

    results = []
    seen_segments: set[str] = set()
    resolution = page_size[0]

    for font_path in font_paths:
        font_name = Path(font_path).stem

        for erosion in (False, True):
            for yellow in (False, True):
                for font_size_label, font_size in font_size_labels.items():
                    for sample_idx in range(sample_num):
                        text_segment = _pick_unique_segment(seen_segments)
                        seen_segments.add(text_segment)

                        puzzle_name = (
                            f"{resolution}_{font_name}_{int(erosion)}_"
                            f"{int(yellow)}_{font_size_label}_{sample_idx}"
                        )
                        output_base = os.path.join(output_dir, puzzle_name)

                        generate_text_puzzle(
                            text_segment,
                            output_base,
                            font_path,
                            page_size=page_size,
                            font_size=font_size,
                            margin=margin,
                            grid_size=grid_size,
                            erosion=erosion,
                            erosion_width=erosion_width if erosion else 0,
                            paper_yellowing=yellow,
                        )
                        results.append(output_base)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text puzzle data from a text folder and font folder."
    )
    parser.add_argument("--txt-dir", required=True, help="Path to a folder containing .txt files")
    parser.add_argument("--font-dir", required=True, help="Path to a folder containing font files (.ttf, .otf, .ttc)")
    parser.add_argument(
        "--segment-chars",
        type=int,
        required=True,
        help="Number of characters in each generated text segment",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        required=True,
        help="Number of samples to generate for each font/variation/font size",
    )
    parser.add_argument(
        "--image-size",
        default="576",
        help="Image resolution: single value or width,height. Default is 576",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write generated puzzles and labels",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=3,
        help="Grid size for each puzzle page. Default is 3",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=1,
        help="Margin around text inside each puzzle piece. Default is 1",
    )
    args = parser.parse_args()

    generated = generate_dataset_from_dirs(
        args.txt_dir,
        args.font_dir,
        args.segment_chars,
        args.sample_num,
        args.output_dir,
        image_size=args.image_size,
        grid_size=args.grid_size,
        margin=args.margin,
    )

    print(f"Generated {len(generated)} puzzle bases in {args.output_dir}")
