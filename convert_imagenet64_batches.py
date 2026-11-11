"""将 ImageNet64 train_data_batch_* 无损转换为当前加载器可读的 PNG。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


Source_dir = Path("data/imagenet64")
Output_dir = Path("data/imagenet64_png")
Workers = 2
Png_compress_level = 1


def _write_png(task: tuple[np.ndarray, Path, int]) -> bool:
    chw_image, target, compress_level = task
    if target.is_file():
        return False
    image = np.transpose(chw_image.reshape(3, 64, 64), (1, 2, 0))
    temporary = target.with_suffix(".png.tmp")
    Image.fromarray(image, mode="RGB").save(
        temporary,
        format="PNG",
        compress_level=compress_level,
    )
    temporary.replace(target)
    return True


def _locate_image_block(source: Path) -> tuple[int, int]:
    """定位协议3 pickle中保存 uint8 图像矩阵的 BINBYTES 数据块。"""

    with source.open("rb") as stream:
        header = stream.read(1024 * 1024)
    file_size = source.stat().st_size
    image_bytes = 3 * 64 * 64
    data_marker = b"X\x04\x00\x00\x00data"
    data_position = header.find(data_marker)
    assert data_position >= 0, f"data field was not found in {source}"
    for index in range(data_position + len(data_marker), len(header) - 4):
        opcode = header[index]
        if opcode != ord("B"):
            continue
        length = int.from_bytes(header[index + 1 : index + 5], "little")
        data_start = index + 5
        if length > 0 and length % image_bytes == 0 and data_start + length <= file_size:
            return data_start, length
    assert False, f"Image data block was not found in {source}"


def convert_batch(
    source: str | Path,
    output: str | Path,
    workers: int = Workers,
    compress_level: int = Png_compress_level,
) -> int:
    """转换一个官方 batch；已存在的 PNG 会直接复用。"""

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    data_start, data_length = _locate_image_block(source_path)
    image_bytes = 3 * 64 * 64
    image_count = data_length // image_bytes

    chunk_size = 2048
    with source_path.open("rb") as stream, ThreadPoolExecutor(
        max_workers=max(1, int(workers))
    ) as executor:
        stream.seek(data_start)
        for start in range(0, image_count, chunk_size):
            stop = min(start + chunk_size, image_count)
            raw = stream.read((stop - start) * image_bytes)
            images = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3, 64, 64)
            tasks = (
                (
                    images[offset],
                    output_path / f"{start + offset:06d}.png",
                    int(compress_level),
                )
                for offset in range(len(images))
            )
            list(executor.map(_write_png, tasks))
            print(
                f"[{source_path.name}] {stop}/{image_count} PNG ready",
                flush=True,
            )
    return image_count


if __name__ == "__main__":
    batch_paths = sorted(
        Source_dir.glob("train_data_batch_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    total = 0
    for batch_path in batch_paths:
        batch_output = Output_dir / batch_path.name
        total += convert_batch(batch_path, batch_output)
    print(f"ImageNet64 conversion complete: {total} lossless PNG files", flush=True)
