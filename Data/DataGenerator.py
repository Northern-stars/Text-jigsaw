import json
import os
import re

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


def generate_text_puzzle(
    page_text: str,
    output_path: str,
    font_path: str,
    page_size=(2480, 3508),
    font_size=32,
    margin=120,
    grid_size=3,
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
        piece_w - 2 * margin
    )

    usable_h = (
        piece_h - 2 * margin
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

    def flush_segment():

        nonlocal current_text
        nonlocal row_idx
        nonlocal col_idx

        page_grid[row_idx][col_idx] = (
            current_text.strip()
        )

        current_text = ""

        col_idx += 1

        if col_idx >= grid_size:

            col_idx = 0
            row_idx += 1

    # ==================================================
    # fill page
    # ==================================================

    for token in tokens:

        if row_idx >= total_page_rows:
            break

        if token == "\n":

            flush_segment()

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
        "pieces": []
    }

    for piece_id in range(
        grid_size * grid_size
    ):

        piece_row = (
            piece_id // grid_size
        )

        piece_col = (
            piece_id % grid_size
        )

        lines = []

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

            if text.strip():
                lines.append(
                    text
                )

        # render piece

        img = Image.new(
            "RGB",
            (piece_w, piece_h),
            "white"
        )

        draw = ImageDraw.Draw(
            img
        )

        y = margin

        for line in lines:

            draw.text(
                (margin, y),
                line,
                fill="black",
                font=font
            )

            y += line_height

        img_path = (
            f"{output_path}_{piece_id}.png"
        )

        img.save(
            img_path
        )

        piece_text ="<SEG>"+ "".join(
            f"{line}<SEG>"
            for line in lines
        )

        labels["pieces"].append(
            {
                "piece_id": piece_id,
                "row": piece_row,
                "col": piece_col,
                "text": piece_text,
                "segments": lines,
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

if __name__=="__main__":
    resolution=576
    test_font_path="Data/Font/cochocibscriptlatinpro.otf"
    test_puzzle_path=f"Data/PuzzleData/test-puzzle-{resolution}"
    test_text=read_txt_content("Data/RawData/18.txt")
    seg_list=split_text_by_max_length(test_text,4500)
    generate_text_puzzle(seg_list[5],test_puzzle_path,test_font_path,(resolution,resolution),font_size=resolution//30,margin=1,grid_size=3)
    print(read_label_text(test_puzzle_path+".json"))
    visualize_puzzle(test_puzzle_path)
