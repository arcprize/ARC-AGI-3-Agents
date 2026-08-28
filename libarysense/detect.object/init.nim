class DetectionViewer(gr.HTML):
    def __init__(
        self,
        value: tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]]]
        | tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]], dict[str, Any]]
        | None = None,
        *,
        label: str | None = None,
        panel_title: str = "Detections",
        list_height: int = 300,
        score_threshold: tuple[float, float] = (0.0, 1.0),
        keypoint_threshold: float = 0.0,
        keypoint_radius: int = 3,
        **kwargs: object,
    ) -> None:
        html_template = (_STATIC_DIR / "template.html").read_text(encoding="utf-8")
        css_template = (_STATIC_DIR / "style.css").read_text(encoding="utf-8")
        js_on_load = (_STATIC_DIR / "script.js").read_text(encoding="utf-8")

        has_label = label is not None
        super().__init__(
            value=value,
            label=label,
            show_label=has_label,
            container=has_label,
            html_template=html_template,
            css_template=css_template,
            js_on_load=js_on_load,
            panel_title=panel_title,
            list_height=list_height,
            score_threshold_min=score_threshold[0],
            score_threshold_max=score_threshold[1],
            keypoint_threshold=keypoint_threshold,
            keypoint_radius=keypoint_radius,
            **kwargs,
        )

    def postprocess(self, value: Any) -> str | None:  # noqa: ANN401
        if isinstance(value, str):
            return value
        return self._process(value)

    def _process(
        self,
        value: tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]]]
        | tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]], dict[str, Any]]
        | None,
    ) -> str | None:
        if value is None:
            return None

        if len(value) == 3:  # noqa: PLR2004 - tuple length check
            image_src, annotations, config = value
        else:
            image_src, annotations = value
            config = {}

        img = _load_image(image_src)
        image_url = _save_image_to_cache(img, self.GRADIO_CACHE)

        processed: list[dict[str, Any]] = []
        for i, ann in enumerate(annotations):
            has_kps = "keypoints" in ann and len(ann["keypoints"]) > 0
            default_label = f"Person {i + 1}" if has_kps else f"Detection {i + 1}"

            entry: dict[str, Any] = {
                "keypoints": ann.get("keypoints", []),
                "connections": ann.get("connections", []),
                "color": ann.get("color", _COLOR_PALETTE[i % len(_COLOR_PALETTE)]),
                "label": ann.get("label", default_label),
            }
            if "bbox" in ann:
                entry["bbox"] = ann["bbox"]
            if "score" in ann:
                entry["score"] = ann["score"]
            if "mask" in ann:
                entry["mask"] = ann["mask"]
            processed.append(entry)

        result: dict[str, Any] = {"image": image_url, "annotations": processed}
        if "score_threshold" in config:
            result["scoreThresholdMin"] = config["score_threshold"][0]
            result["scoreThresholdMax"] = config["score_threshold"][1]

        return json.dumps(result)

    def process_example(self, value: Any) -> str | None:  # noqa: ANN401
        if value is None:
            return None
        image_source = value[0] if isinstance(value, tuple) else value
        img = _load_image(image_source)
        url = _save_image_to_cache(img, self.GRADIO_CACHE)
        return (
            f'<img src="{url}" alt="example" '
            f'style="max-width:100%;max-height:5rem;object-fit:contain;'
            f'display:block;border-radius:4px;">'
        )

    def api_info(self) -> dict[str, Any]:
        return {
            "type": "string",
            "description": (
                "JSON string containing detection visualization data. "
                "Structure: {image: string (URL), annotations: ["
                "{color: string, label: string, "
                "bbox?: {x: float, y: float, width: float, height: float}, "
                "score?: float, "
                "mask?: {counts: [int], size: [int, int]}, "
                "keypoints?: [{x: float, y: float, name: string, confidence?: float}], "
                "connections?: [[int, int]]}], "
                "scoreThresholdMin?: float, scoreThresholdMax?: float}"
            ),
        }
