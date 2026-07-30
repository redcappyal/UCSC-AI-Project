from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "index.html"
).read_text(encoding="utf-8")


def test_web_clip_analysis_defaults_to_every_frame():
    assert "frameStride:1" in INDEX_HTML
    assert "frameStride:4" not in INDEX_HTML
    assert "form.append('frame_stride', S.clip.frameStride.toString())" in INDEX_HTML
