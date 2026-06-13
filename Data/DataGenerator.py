import json
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


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
    Parameters
    ----------
    page_text : str

    output_path : str
        例如:
        ./dataset/page_0001

    font_path : str
        ttf字体路径
    """

    page_w, page_h = page_size

    img = Image.new(
        "RGB",
        (page_w, page_h),
        "white"
    )

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(
        font_path,
        font_size
    )

    line_height = int(font_size * 1.5)

    x = margin
    y = margin

    line_id = 0
    char_id = 0

    char_records = []

    for ch in page_text:

        if ch == "\n":
            x = margin
            y += line_height
            line_id += 1
            continue

        bbox = draw.textbbox(
            (x, y),
            ch,
            font=font
        )

        char_w = bbox[2] - bbox[0]

        if x + char_w > page_w - margin:

            x = margin
            y += line_height
            line_id += 1

            bbox = draw.textbbox(
                (x, y),
                ch,
                font=font
            )

            char_w = bbox[2] - bbox[0]

        if y + line_height > page_h - margin:
            break

        draw.text(
            (x, y),
            ch,
            fill="black",
            font=font
        )

        char_records.append(
            {
                "char_id": char_id,
                "char": ch,
                "line_id": line_id,

                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
            }
        )

        x += char_w
        char_id += 1

    piece_w = page_w // grid_size
    piece_h = page_h // grid_size

    labels = {
        "grid_size": grid_size,
        "page_width": page_w,
        "page_height": page_h,
        "pieces": []
    }

    for row in range(grid_size):

        for col in range(grid_size):

            piece_id = row * grid_size + col

            left = col * piece_w
            top = row * piece_h

            right = (
                page_w
                if col == grid_size - 1
                else (col + 1) * piece_w
            )

            bottom = (
                page_h
                if row == grid_size - 1
                else (row + 1) * piece_h
            )

            crop = img.crop(
                (left, top, right, bottom)
            )

            crop.save(
                f"{output_path}_{piece_id}.png"
            )

            piece_records = []

            for rec in char_records:

                cx = (rec["x1"] + rec["x2"]) / 2
                cy = (rec["y1"] + rec["y2"]) / 2

                if (
                    left <= cx < right
                    and
                    top <= cy < bottom
                ):
                    piece_records.append(
                        {
                            "char_id": rec["char_id"],
                            "char": rec["char"],
                            "line_id": rec["line_id"]
                        }
                    )

            piece_records.sort(
                key=lambda x: x["char_id"]
            )

            segments = []
            current_segment = []

            prev_char_id = None

            for rec in piece_records:

                if (
                    prev_char_id is not None
                    and
                    rec["char_id"] != prev_char_id + 1
                ):

                    if current_segment:
                        segments.append(
                            "".join(current_segment)
                        )

                    current_segment = []

                current_segment.append(
                    rec["char"]
                )

                prev_char_id = rec["char_id"]

            if current_segment:
                segments.append(
                    "".join(current_segment)
                )

            piece_text = " <SEG> ".join(
                seg.strip()
                for seg in segments
                if seg.strip()
            )

            labels["pieces"].append(
                {
                    "piece_id": piece_id,
                    "row": row,
                    "col": col,

                    "text": piece_text,
                    "segments": segments,

                    "char_ids": [
                        r["char_id"]
                        for r in piece_records
                    ],

                    "image":
                    os.path.basename(
                        f"{output_path}_{piece_id}.png"
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
            f"=== Piece {piece['piece_id']} ==="
        )

        output.append(
            piece["text"]
        )

        output.append("")

    return "\n".join(output)