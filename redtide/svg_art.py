"""Generate safe SVG artwork from a report using the configured chat model."""
import os
import re
import xml.etree.ElementTree as ET

from . import llm

# 插图默认走轻量免思考模型：GLM-5.3-Flash 对"画信息图"会持续思考，
# 即使 effort=low 也曾把 8000 tokens 全部耗在思考上、正文为空（踩坑实录）。
# 可用环境变量 SVG_MODEL 覆盖。
SVG_MODEL = os.getenv("SVG_MODEL", "glm-4-flash")


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
SVG_TAGS = {
    "svg", "g", "rect", "circle", "ellipse", "line", "polyline",
    "polygon", "path", "text", "tspan",
}
SVG_ATTRIBUTES = {
    "viewBox", "width", "height", "x", "y", "x1", "x2", "y1", "y2",
    "cx", "cy", "r", "rx", "ry", "d", "points", "fill", "stroke",
    "stroke-width", "stroke-linecap", "stroke-linejoin", "fill-opacity",
    "stroke-opacity", "opacity", "font-size", "font-family", "font-weight",
    "text-anchor", "dominant-baseline", "dx", "dy", "transform",
}
MAX_SVG_LENGTH = 50_000
MAX_SVG_ELEMENTS = 500
MAX_SVG_DEPTH = 20


class SVGGenerationError(ValueError):
    """Raised when model output is not a safe, renderable SVG illustration."""


def sanitize_svg(source: str) -> str:
    """Validate and serialize SVG markup using a deliberately small allowlist."""
    source = source.strip()
    fenced = re.fullmatch(r"```(?:svg|xml)?\s*(.*?)\s*```", source, re.DOTALL | re.IGNORECASE)
    if fenced:
        source = fenced.group(1).strip()
    if not source or len(source) > MAX_SVG_LENGTH:
        raise SVGGenerationError("模型没有返回有效长度的 SVG。")
    if "<!doctype" in source.lower() or "<!entity" in source.lower():
        raise SVGGenerationError("SVG 不允许包含 DOCTYPE 或 ENTITY 声明。")

    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise SVGGenerationError("模型返回的 SVG 格式无效。") from exc

    def normalize(element: ET.Element, depth: int) -> int:
        if depth > MAX_SVG_DEPTH:
            raise SVGGenerationError("SVG 嵌套层级超过限制。")

        if element.tag.startswith("{"):
            namespace, local_name = element.tag[1:].split("}", 1)
            if namespace != SVG_NAMESPACE:
                raise SVGGenerationError("SVG 中包含不支持的 XML 命名空间。")
        else:
            local_name = element.tag
        if local_name not in SVG_TAGS:
            raise SVGGenerationError(f"SVG 中包含不支持的元素：{local_name}")
        element.tag = f"{{{SVG_NAMESPACE}}}{local_name}"

        for name in list(element.attrib):
            value = element.attrib[name]
            if name.startswith("{") or name not in SVG_ATTRIBUTES:
                del element.attrib[name]
                continue
            if len(value) > 2_000 or "url(" in value.casefold():
                raise SVGGenerationError("SVG 属性包含过长内容或外部资源引用。")

        count = 1
        for child in element:
            count += normalize(child, depth + 1)
            if count > MAX_SVG_ELEMENTS:
                raise SVGGenerationError("SVG 元素数量超过限制。")
        return count

    if root.tag not in ("svg", f"{{{SVG_NAMESPACE}}}svg"):
        raise SVGGenerationError("模型输出根元素不是 SVG。")
    normalize(root, 1)
    root.set("width", root.get("width", "800"))
    root.set("height", root.get("height", "450"))
    root.set("viewBox", root.get("viewBox", "0 0 800 450"))
    ET.register_namespace("", SVG_NAMESPACE)
    return ET.tostring(root, encoding="unicode")


def generate_report_svg(question: str, report: str) -> str:
    """Ask the configured chat model for SVG markup illustrating a report."""
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Create a simple, readable scientific infographic as SVG markup. "
                    "Use only SVG vector elements: svg, g, rect, circle, ellipse, line, "
                    "polyline, polygon, path, text, and tspan. Use a viewBox of "
                    "'0 0 800 450'. Do not use scripts, styles, links, images, filters, "
                    "foreignObject, or external resources. Do not invent scientific facts "
                    "not present in the supplied report. Return only the SVG document."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question[:1000]}\n\nReport:\n{report[:5000]}",
            },
        ],
        temperature=0.2,
        # 思考型模型思考+SVG 代码都占输出预算，预算太小正文会被截成空
        max_tokens=4000,
        model=SVG_MODEL,
    )
    return sanitize_svg(response.content)
