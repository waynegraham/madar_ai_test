from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
NS = {"alto": ALTO_NS}


@dataclass
class AltoLine:
    id: str
    text: str
    bbox: tuple[float, float, float, float]
    polygon: list[tuple[float, float]] = field(default_factory=list)
    baseline: list[tuple[float, float]] = field(default_factory=list)
    confidence: Optional[float] = None
    tagrefs: list[str] = field(default_factory=list)


@dataclass
class AltoBlock:
    id: str
    block_type: Optional[str]
    bbox: Optional[tuple[float, float, float, float]]
    polygon: list[tuple[float, float]] = field(default_factory=list)
    lines: list[AltoLine] = field(default_factory=list)
    tagrefs: list[str] = field(default_factory=list)


@dataclass
class AltoPage:
    image_filename: Optional[str]
    width: int
    height: int
    physical_image_number: Optional[int]
    blocks: list[AltoBlock]
    tags: dict[str, str] = field(default_factory=dict)


def _parse_points(value: Optional[str]) -> list[tuple[float, float]]:
    """
    Convert an ALTO coordinate string such as:

        "1365 766 1637 833 1637 1519"

    into:

        [(1365, 766), (1637, 833), (1637, 1519)]
    """
    if not value:
        return []

    numbers = [float(n) for n in value.split()]

    if len(numbers) % 2:
        raise ValueError(f"Invalid coordinate list: {value}")

    return list(zip(numbers[0::2], numbers[1::2]))


def _parse_bbox(
    element: ET.Element,
) -> Optional[tuple[float, float, float, float]]:
    """
    Convert ALTO HPOS/VPOS/WIDTH/HEIGHT into:

        (x1, y1, x2, y2)
    """
    required = ("HPOS", "VPOS", "WIDTH", "HEIGHT")

    if not all(element.get(attr) is not None for attr in required):
        return None

    x = float(element.get("HPOS"))
    y = float(element.get("VPOS"))
    width = float(element.get("WIDTH"))
    height = float(element.get("HEIGHT"))

    return x, y, x + width, y + height


def _parse_shape(element: ET.Element) -> list[tuple[float, float]]:
    polygon = element.find("alto:Shape/alto:Polygon", NS)

    if polygon is None:
        return []

    return _parse_points(polygon.get("POINTS"))


def _parse_tagrefs(element: ET.Element) -> list[str]:
    value = element.get("TAGREFS")

    if not value:
        return []

    return value.split()


def _parse_tags(root: ET.Element) -> dict[str, str]:
    """
    Return mappings such as:

        {
            "BT1": "Title",
            "BT2": "Main",
            "BT3": "Commentary",
            "BT26346": "Stamp",
            "BT26347": "Note",
        }
    """
    tags: dict[str, str] = {}

    for tag in root.findall(".//alto:Tags/*", NS):
        tag_id = tag.get("ID")
        label = tag.get("LABEL")

        if tag_id and label:
            tags[tag_id] = label

    return tags


def _parse_line(element: ET.Element) -> AltoLine:
    strings = element.findall("alto:String", NS)

    # ALTO can theoretically contain multiple String elements in a
    # TextLine. eScriptorium usually exports one, but joining them
    # makes the parser more general.
    text_parts: list[str] = []
    confidences: list[float] = []

    for string in strings:
        content = string.get("CONTENT")

        if content:
            text_parts.append(content)

        wc = string.get("WC")
        if wc:
            try:
                confidences.append(float(wc))
            except ValueError:
                pass

    confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    return AltoLine(
        id=element.get("ID", ""),
        text=" ".join(text_parts),
        bbox=_parse_bbox(element) or (0, 0, 0, 0),
        polygon=_parse_shape(element),
        baseline=_parse_points(element.get("BASELINE")),
        confidence=confidence,
        tagrefs=_parse_tagrefs(element),
    )


def _parse_block(
    element: ET.Element,
    tags: dict[str, str],
) -> AltoBlock:

    tagrefs = _parse_tagrefs(element)

    block_type = None

    for tagref in tagrefs:
        if tagref in tags:
            block_type = tags[tagref]
            break

    lines = [
        _parse_line(line)
        for line in element.findall("alto:TextLine", NS)
    ]

    return AltoBlock(
        id=element.get("ID", ""),
        block_type=block_type,
        bbox=_parse_bbox(element),
        polygon=_parse_shape(element),
        lines=lines,
        tagrefs=tagrefs,
    )


def load_alto(path: str | Path) -> AltoPage:
    """
    Parse an eScriptorium ALTO XML export.

    Example:

        page = load_alto("data/alto/436_AB008999_0003.xml")

        print(page.image_filename)

        for block in page.blocks:
            print(block.block_type)

            for line in block.lines:
                print(line.text)
    """

    path = Path(path)

    tree = ET.parse(path)
    root = tree.getroot()

    tags = _parse_tags(root)

    page_element = root.find(".//alto:Page", NS)

    if page_element is None:
        raise ValueError(f"No ALTO Page found in {path}")

    source_filename = root.findtext(
        ".//alto:sourceImageInformation/alto:fileName",
        default=None,
        namespaces=NS,
    )

    blocks = [
        _parse_block(block, tags)
        for block in page_element.findall(
            ".//alto:PrintSpace/alto:TextBlock",
            NS,
        )
    ]

    physical_number = page_element.get("PHYSICAL_IMG_NR")

    return AltoPage(
        image_filename=source_filename,
        width=int(float(page_element.get("WIDTH", "0"))),
        height=int(float(page_element.get("HEIGHT", "0"))),
        physical_image_number=(
            int(physical_number)
            if physical_number is not None
            else None
        ),
        blocks=blocks,
        tags=tags,
    )