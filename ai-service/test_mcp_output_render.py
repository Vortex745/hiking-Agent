"""Unit contract for composing custom-MCP outputs into chat content."""

from __future__ import annotations

import json

from agent_v2.mcp_output_render import (
    compose_mcp_reply,
    extract_image_urls,
    extract_mcp_images,
    render_mcp_outputs,
)

_BIG = "https://cdn.example.com/big/white-cloud.jpg"
_PREVIEW = "https://preview.example.com/white-cloud.jpg!qt_kuan320"
_ORIGIN = "https://origin.example.com/white-cloud.jpg?token=protected"
_DOWNLOAD = "https://origin.example.com/download/white-cloud.jpg"


def _output(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _mountain(**overrides):
    obj = {
        "pid": 101,
        "title": "白云山",
        "big_url": _BIG,
        "preview_url": _PREVIEW,
        "origin_url": _ORIGIN,
        "download_url": _DOWNLOAD,
    }
    obj.update(overrides)
    return obj


def test_one_object_with_versions_renders_one_public_preview():
    rendered = render_mcp_outputs([_output({"ok": True, "data": {"list": [_mountain()]}})])
    assert rendered == f"![白云山]({_BIG})"
    assert "共 1 张图片" not in rendered
    assert _ORIGIN not in rendered
    assert _DOWNLOAD not in rendered
    assert "!qt_kuan320" not in rendered


def test_display_priority_falls_back_to_preview_when_big_is_missing():
    mountain = _mountain()
    mountain.pop("big_url")
    images = extract_mcp_images([_output({"data": {"list": [mountain]}})])
    assert [image.display_url for image in images] == [_PREVIEW]
    assert images[0].origin_urls == {_ORIGIN, _DOWNLOAD}


def test_model_images_are_rewritten_to_the_selected_preview_once():
    model = (
        f"白云山：![白云山]({_PREVIEW})\n"
        f"再来一次：![白云山]({_ORIGIN})\n"
        f"原图在这里：[原图]({_ORIGIN})，官网：[官网](https://example.com/page)。"
    )
    composed = compose_mcp_reply(model, [("svc-a", _output({"data": {"list": [_mountain()]}}))])
    assert composed.count("![") == 1
    assert f"![白云山]({_BIG})" in composed
    assert f"[原图]({_BIG})" in composed
    assert "[官网](https://example.com/page)" in composed
    assert _ORIGIN not in composed
    assert _PREVIEW not in composed
    assert "共 1 张图片" not in composed


def test_preview_links_render_inline_without_appending_the_result_list():
    first = _mountain()
    second = {
        "pid": 102,
        "title": "另一张",
        "preview_url": "https://preview.example.com/other.jpg!qt_kuan320",
    }
    composed = compose_mcp_reply(
        f"1. 白云山\n   [预览图]({_PREVIEW})",
        [("svc-a", _output({"data": {"list": [first, second]}}))],
    )
    assert composed == f"1. 白云山\n   ![预览图]({_BIG})"
    assert "other.jpg" not in composed


def test_fallback_appends_at_most_three_previews():
    objects = [
        {"pid": index, "preview_url": f"https://cdn.example.com/{index}.jpg"}
        for index in range(5)
    ]
    composed = compose_mcp_reply("结果", [("svc-a", _output({"data": {"list": objects}}))])
    assert composed.count("![") == 3


def test_missing_image_is_appended_without_a_count_tail():
    composed = compose_mcp_reply(
        "我找到了白云山的照片。",
        [("svc-a", _output({"data": {"list": [_mountain()]}}))],
    )
    assert composed.startswith("我找到了白云山的照片。")
    assert composed.count("![") == 1
    assert f"![白云山]({_BIG})" in composed
    assert "共 1 张图片" not in composed


def test_same_object_across_queries_is_deduplicated_within_one_service():
    first = _output({"data": {"list": [_mountain()]}})
    second = _output({"data": {"list": [{"pid": 101, "title": "白云山", "preview_url": _PREVIEW}]}})
    outputs = [("svc-a", first), ("svc-a", second)]
    images = extract_mcp_images(outputs)
    assert len(images) == 1
    assert images[0].display_url == _BIG
    composed = compose_mcp_reply("结果", outputs)
    assert composed.count("![") == 1


def test_same_identifier_in_different_services_is_not_merged():
    one = _output({"data": {"list": [{"pid": 101, "preview_url": "https://a.example.com/1.jpg"}]}})
    two = _output({"data": {"list": [{"pid": 101, "preview_url": "https://b.example.com/2.jpg"}]}})
    images = extract_mcp_images([("svc-a", one), ("svc-b", two)])
    assert len(images) == 2
    assert {image.service for image in images} == {"svc-a", "svc-b"}


def test_unidentified_objects_merge_only_on_an_exact_alias():
    same_a = {"data": {"list": [{"preview_url": "https://cdn.example.com/same.jpg"}]}}
    same_b = {"data": {"list": [{"preview_url": "https://cdn.example.com/same.jpg"}]}}
    other = {"data": {"list": [{"preview_url": "https://cdn.example.com/other.jpg"}]}}
    images = extract_mcp_images([_output(same_a), _output(same_b), _output(other)])
    assert [image.display_url for image in images] == [
        "https://cdn.example.com/same.jpg",
        "https://cdn.example.com/other.jpg",
    ]


def test_origin_only_objects_are_known_but_never_auto_previewed():
    origin_only = {"data": {"list": [{"pid": 9, "title": "原图", "origin_url": _ORIGIN}]}}
    rendered = render_mcp_outputs([_output(origin_only)])
    assert rendered == ""
    composed = compose_mcp_reply(f"查看 [原图]({_ORIGIN})", [("svc-a", _output(origin_only))])
    assert composed == f"查看 [原图]({_ORIGIN})"
    assert "![" not in composed


def test_generate_image_shapes_are_recognized():
    shapes = [
        {"data": {"images": [{"url": "https://cdn.example.com/a.png", "title": "A"}]}},
        {"data": {"image_urls": ["https://cdn.example.com/b.jpg"]}},
        {"data": {"list": [{"url": "https://cdn.example.com/c.webp"}]}},
    ]
    for payload in shapes:
        images = extract_mcp_images([_output(payload)])
        assert len(images) == 1, payload


def test_text_only_output_is_not_appended():
    output = _output({"ok": True, "data": {"credits_remaining": 0, "free_points": 10}})
    assert extract_image_urls(output) == []
    assert render_mcp_outputs([output]) == ""


def test_plain_text_output_is_kept_but_json_is_dropped():
    assert render_mcp_outputs(["echo:hello:walk"]) == "echo:hello:walk"
    json_only = _output({"ok": True, "data": {"free_points": 10}})
    assert render_mcp_outputs([json_only]) == ""


def test_non_http_duplicates_and_cap_are_filtered():
    objects = [
        {"pid": index, "preview_url": f"https://cdn.example.com/{index}.png"}
        for index in range(15)
    ]
    payload = {
        "images": [
            *objects,
            {"pid": 0, "preview_url": "https://cdn.example.com/0.png"},
            {"preview_url": "javascript:alert(1)"},
            {"preview_url": "data:image/png;base64,AAAA"},
            {"preview_url": "file:///C:/a.png"},
            {"preview_url": "not-a-url"},
        ],
    }
    images = extract_mcp_images([_output(payload)])
    assert len(images) == 12
    assert len({image.display_url for image in images}) == 12
    assert all(image.display_url.startswith("https://") for image in images)
    rendered = render_mcp_outputs([_output(payload)])
    assert rendered.count("![") == 12


def test_url_key_without_an_image_extension_is_not_an_image():
    payload = {"data": {"url": "https://api.example.com/v1/tasks/1"}}
    assert extract_image_urls(_output(payload)) == []


def test_markdown_image_in_plain_output_is_preserved():
    plain = "结果：![预览](https://cdn.example.com/a.png)"
    images = extract_image_urls(plain)
    assert images == [("https://cdn.example.com/a.png", "预览")]
    rendered = render_mcp_outputs([plain])
    assert rendered == "![预览](https://cdn.example.com/a.png)"


def test_prior_text_prevents_reappending_after_resume():
    output = _output({"data": {"list": [_mountain()]}})
    composed = compose_mcp_reply(
        "继续说明。", [("svc-a", output)], prior_text=f"![白云山]({_BIG})"
    )
    assert composed == "继续说明。"


def test_query_parameters_and_cdn_suffixes_are_preserved_exactly():
    url = "https://cdn.example.com/a.jpg?x-oss-process=image/resize,w_320!qt_kuan320&sign=abc"
    payload = {"data": {"list": [{"pid": 1, "preview_url": url}]}}
    composed = compose_mcp_reply("结果", [("svc-a", _output(payload))])
    assert f"]({url})" in composed
