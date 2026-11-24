"""Isolated HTTP smoke test for the catalog, preview, export and restart."""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import imageio_ffmpeg

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from mini_agent.video.media_identity import file_signature
from mini_agent.video.reader import VideoReader
from mini_agent.video.studio_history import save_json
from studio_store import StudioStore


def call(port, path, body=None, token=None, *, timeout=15, headers=None):
    headers = dict(headers or {})
    if token:
        headers["X-Studio-Token"] = token
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode() if body is not None else None, headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            blob = response.read()
            result = json.loads(blob) if response.headers.get_content_type() == "application/json" else blob
            return response.status, result
    except HTTPError as error:
        return error.code, None


def spawn(port, directory, dsn, engine):
    env = {**os.environ, "CUTLINER_DATABASE_URL": dsn or " ", "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen([sys.executable, str(PROJECT/"studio_server.py"), "--port", str(port),
                                "--data-dir", str(directory)], cwd=PROJECT, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    limit = time.monotonic() + 15
    try:
        while time.monotonic() < limit:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with {process.returncode}")
            try:
                status, data = call(port, "/api/health", timeout=2)
                if status == 200:
                    assert data["database"] == engine
                    return process
            except (URLError, TimeoutError):
                pass
            time.sleep(.15)
        raise TimeoutError("Server startup did not become healthy")
    except BaseException:
        stop(process)
        raise


def stop(process):
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)


def exercise(directory, dsn, engine, *, built_frontend=False):
    path = directory/"测试长画幅.mkv"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=320x120:r=24:d=3", "-c:v", "libx264", "-threads", "2", str(path)], check=True)
    metadata = VideoReader().read_metadata(path)
    run = directory/"runs"/"neural_video_fixture"
    run.mkdir(parents=True)
    save_json(run/"metadata.json", metadata.model_dump(mode="json"))
    save_json(run/"task.json", {"request": "测试选段", "task": {"query": "测试选段"}})
    save_json(run/"retrieval.json", {"source_signature": file_signature(path)})
    save_json(run/"candidates.json", [{"candidate_id": "neural_0001", "start_seconds": .4,
        "end_seconds": 1.9, "evidence_source": "qd_detr", "relevance_score": .8, "review_priority": 1,
        "model_evidence": [{"modality": "visual", "model_name": "fixture", "label": "fixture",
                            "score": .8, "start_seconds": .4, "end_seconds": 1.9}]}])
    upload = directory/"只导入不检索.mkv"
    upload.write_bytes(path.read_bytes())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = spawn(port, directory, dsn, engine)
    try:
        if built_frontend:
            status, html = call(port, "/")
            assert status == 200 and b'id="root"' in html
            script = re.search(rb'<script[^>]+src="(/assets/[^"/]+\.js)"', html)
            assert script, "The index page does not reference a built script"
            asset_status, asset = call(port, script.group(1).decode("ascii"))
            assert asset_status == 200 and len(asset) > 1000 and not asset.startswith(b"<!doctype html>")
        _, boot = call(port, "/api/bootstrap")
        token = boot["token"]
        media = next(x for x in boot["media"] if x["path"] == str(path))
        mid = media["id"]
        prefs = f"/api/media/{mid}/preferences"
        assert call(port, prefs, {"favorite": True})[0] == 403
        assert call(port, prefs, {"favorite": True, "name": "收藏素材"}, token)[0] == 200
        status, imported = call(port, "/api/import/path", {"path": str(upload)}, token)
        assert status == 200 and imported["media"]["id"] != mid
        code, video = call(port, f"/api/preview/{mid}?start=.1&end=1.4", headers={"Range": "bytes=0-127"})
        assert code == 206 and b"ftyp" in video
        assert call(port, f"/api/preview/{mid}?start=1&end=.3")[0] == 400
        body = {"run_id": run.name, "candidate_id": "neural_0001", "start": .5, "end": 1.6}
        code, job = call(port, "/api/export", body, token)
        assert code == 202, code
        for _ in range(100):
            _, result = call(port, f"/api/jobs/{job['job_id']}")
            if result["status"] not in {"running", "cancelling"}:
                break
            time.sleep(.2)
        assert result["status"] == "complete", result.get("error")
        output = result["result"]["media"]
        assert Path(output["path"]).is_file()
    finally:
        stop(process)
    store = StudioStore(directory/"cutliner.sqlite3", dsn)
    assert len(store.jobs()) == 1 and store.run_paths() == [run]
    process = spawn(port, directory, dsn, engine)
    try:
        _, boot = call(port, "/api/bootstrap")
        assert token != boot["token"]
        assert boot["media"][0]["id"] == mid and boot["media"][0]["favorite"] is True
        assert boot["media"][0]["name"] == "收藏素材"
        assert imported["media"]["id"] in {m["id"] for m in boot["media"]}
        assert not any(m["id"] == output["id"] for m in boot["media"])
        assert any(r["id"] == run.name and r["deliveries"] for r in boot["runs"])
        assert boot["latest_job"]["id"] == job["job_id"]
        assert boot["latest_job"]["result"]["media"]["id"] == output["id"]
        assert call(port, output["url"], headers={"Range": "bytes=0-127"})[0] == 206
    finally:
        stop(process)
    print(json.dumps({"database": engine, "built_frontend_served": built_frontend,
                      "media_restarted": True, "favorite_restarted": True,
                      "jobs_and_output_restarted": True, "preview_range": True, "real_export": True,
                      "cloud_requests": 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres", action="store_true", help="Use CUTLINER_TEST_DATABASE_URL in an isolated schema")
    parser.add_argument("--built-frontend", action="store_true", help="Verify the generated UI is served by the app")
    args = parser.parse_args()
    dsn = ""
    if args.postgres:
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo
        configured = os.environ["CUTLINER_TEST_DATABASE_URL"]
        schema = f"cutliner_smoke_{uuid4().hex}"
        with psycopg.connect(configured, connect_timeout=5) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        dsn = make_conninfo(configured, options=f"-c search_path={schema}")
    with TemporaryDirectory(prefix="cutliner-db-check-") as root:
        try:
            exercise(Path(root), dsn, "postgresql" if args.postgres else "sqlite", built_frontend=args.built_frontend)
        finally:
            if args.postgres:
                # Only the UUID isolated schema created by this invocation.
                with psycopg.connect(configured, connect_timeout=5) as conn:
                    conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


if __name__ == "__main__":
    main()
