"""Compact image encoding used only for multimodal feedback transport."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


_SOURCE_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True)
class TransportImage:
    payload: bytes
    mime_type: str
    original_bytes: int
    original_size: tuple[int, int]
    transmitted_size: tuple[int, int]


def _rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def encode_transport_image(
    image_path: str | Path,
    *,
    compress: bool = True,
    max_side: int = 1024,
    quality: int = 80,
) -> TransportImage:
    """Return the original image or a smaller transport-only JPEG copy."""
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    if not 1 <= quality <= 95:
        raise ValueError("quality must be between 1 and 95")

    path = Path(image_path).expanduser().resolve()
    source_payload = path.read_bytes()
    with Image.open(path) as source:
        source.load()
        source_format = (source.format or "").upper()
        oriented = ImageOps.exif_transpose(source)
        original_size = oriented.size
        source_mime = _SOURCE_MIME_TYPES.get(source_format)
        if not compress and source_mime:
            return TransportImage(
                payload=source_payload,
                mime_type=source_mime,
                original_bytes=len(source_payload),
                original_size=original_size,
                transmitted_size=original_size,
            )
        compact = _rgb_image(oriented)

    if not compress:
        output = io.BytesIO()
        compact.save(output, format="PNG")
        return TransportImage(
            payload=output.getvalue(),
            mime_type="image/png",
            original_bytes=len(source_payload),
            original_size=original_size,
            transmitted_size=original_size,
        )

    compact.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    compact.save(
        output,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )
    jpeg_payload = output.getvalue()

    if source_mime and len(source_payload) <= len(jpeg_payload):
        return TransportImage(
            payload=source_payload,
            mime_type=source_mime,
            original_bytes=len(source_payload),
            original_size=original_size,
            transmitted_size=original_size,
        )
    return TransportImage(
        payload=jpeg_payload,
        mime_type="image/jpeg",
        original_bytes=len(source_payload),
        original_size=original_size,
        transmitted_size=compact.size,
    )
