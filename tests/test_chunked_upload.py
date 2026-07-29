"""Chunked upload: ingesting a match, not just a clip.

`POST /api/upload` is a whole-file multipart POST capped at 2 GB, which is
about five minutes of the app's own 4K60 capture. A forty-minute match cannot
be ingested at all -- which makes "record a session and analyze it" false for
real sessions, and no amount of analysis downstream fixes that.

The assertions that matter here are about *integrity*, not throughput. A video
that reassembles wrong is worse than one that fails to upload: it decodes,
analyses, and produces statistics about corrupted frames.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client(runs_dir):
    import app as app_module

    return app_module.app.test_client()


def _init(client, filename="match.mp4", size=1024):
    return client.post("/api/upload/init", json={"filename": filename, "size": size})


def test_a_three_chunk_upload_reassembles_byte_identical(runs_dir):
    """The whole point. A file that reassembles wrong still decodes."""
    client = _client(runs_dir)
    payload = bytes(range(256)) * 12
    parts = [payload[:1000], payload[1000:2000], payload[2000:]]

    upload_id = _init(client, size=len(payload)).get_json()["upload_id"]
    for index, part in enumerate(parts):
        response = client.post(
            f"/api/upload/chunk/{upload_id}?index={index}",
            data=part, content_type="application/octet-stream",
        )
        assert response.status_code == 200, response.get_json()

    body = client.post(f"/api/upload/complete/{upload_id}").get_json()

    assert body["ok"] is True
    assert body["video_id"] == hashlib.sha256(payload).hexdigest()

    import app as app_module
    stored = app_module.video_path_for_id(body["video_id"])
    assert stored is not None, "assembled file is not resolvable by video_id"
    assert stored.read_bytes() == payload


def test_the_assembled_file_keeps_an_extension_so_it_can_be_resolved(runs_dir):
    """video_path_for_id globs `<id>.*`, so an extensionless file is lost.

    It would upload successfully, hash correctly, and then be unfindable --
    a failure that only shows up one screen later at TRACK.
    """
    client = _client(runs_dir)
    upload_id = _init(client, filename="match.mov", size=4).get_json()["upload_id"]
    client.post(f"/api/upload/chunk/{upload_id}?index=0", data=b"data",
                content_type="application/octet-stream")

    body = client.post(f"/api/upload/complete/{upload_id}").get_json()

    import app as app_module
    stored = app_module.video_path_for_id(body["video_id"])
    assert stored is not None
    assert stored.suffix == ".mov"


def test_a_chunk_arriving_out_of_order_is_refused(runs_dir):
    """Silently accepting a gap assembles a file with a hole in it.

    That file decodes. It produces statistics. Nothing downstream can tell.
    """
    client = _client(runs_dir)
    upload_id = _init(client).get_json()["upload_id"]
    client.post(f"/api/upload/chunk/{upload_id}?index=0", data=b"aaa",
                content_type="application/octet-stream")

    response = client.post(f"/api/upload/chunk/{upload_id}?index=2", data=b"ccc",
                           content_type="application/octet-stream")

    assert response.status_code == 409


def test_a_repeated_chunk_is_refused_rather_than_appended(runs_dir):
    """A client retry after a timeout must not duplicate the bytes."""
    client = _client(runs_dir)
    upload_id = _init(client).get_json()["upload_id"]
    client.post(f"/api/upload/chunk/{upload_id}?index=0", data=b"aaa",
                content_type="application/octet-stream")

    response = client.post(f"/api/upload/chunk/{upload_id}?index=0", data=b"aaa",
                           content_type="application/octet-stream")

    assert response.status_code == 409


def test_an_oversized_chunk_is_refused(runs_dir):
    import app as app_module

    client = _client(runs_dir)
    upload_id = _init(client).get_json()["upload_id"]

    response = client.post(
        f"/api/upload/chunk/{upload_id}?index=0",
        data=b"x" * (app_module.UPLOAD_CHUNK_MAX_BYTES + 1),
        content_type="application/octet-stream",
    )

    assert response.status_code == 413


def test_chunks_for_an_unknown_upload_404(runs_dir):
    client = _client(runs_dir)

    response = client.post("/api/upload/chunk/nope?index=0", data=b"x",
                           content_type="application/octet-stream")

    assert response.status_code == 404


def test_partials_land_under_the_patched_runs_tree(runs_dir):
    """The partial directory must follow BY_HASH_DIR, not a module constant.

    A module-level absolute path would write real files into the developer's
    upload store during the test run -- the exact class of shared state the
    runs_dir fixture exists to remove.
    """
    client = _client(runs_dir)
    upload_id = _init(client).get_json()["upload_id"]
    client.post(f"/api/upload/chunk/{upload_id}?index=0", data=b"abc",
                content_type="application/octet-stream")

    partials = list(runs_dir.rglob(f"{upload_id}*"))
    assert partials, "partial file was not written under the test runs tree"


def test_completing_an_upload_clears_its_partial(runs_dir):
    """Abandoned partials of a 40-minute match are not small."""
    client = _client(runs_dir)
    upload_id = _init(client).get_json()["upload_id"]
    client.post(f"/api/upload/chunk/{upload_id}?index=0", data=b"abc",
                content_type="application/octet-stream")

    client.post(f"/api/upload/complete/{upload_id}")

    assert not list(runs_dir.rglob(f"{upload_id}.part"))


def test_completing_an_upload_that_never_started_404s(runs_dir):
    client = _client(runs_dir)

    assert client.post("/api/upload/complete/nope").status_code == 404


def test_an_identical_file_dedupes_to_the_same_video_id(runs_dir):
    """Same content-addressed store as /api/upload, so re-uploading is free."""
    client = _client(runs_dir)
    payload = b"identical bytes"

    ids = []
    for _ in range(2):
        upload_id = _init(client, size=len(payload)).get_json()["upload_id"]
        client.post(f"/api/upload/chunk/{upload_id}?index=0", data=payload,
                    content_type="application/octet-stream")
        ids.append(client.post(f"/api/upload/complete/{upload_id}").get_json()["video_id"])

    assert ids[0] == ids[1]
    import app as app_module
    assert len(list(app_module.BY_HASH_DIR.glob(f"{ids[0]}.*"))) == 1
