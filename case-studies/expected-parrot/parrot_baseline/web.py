"""Tiny stdlib web app: a form, a POST /run that executes the demo engine, and a whitelist file server."""

from __future__ import annotations

import argparse
import ipaddress
import re
import secrets
import threading
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .engines import DemoEngine
from .report import LABEL
from .runner import ARTIFACTS, run_study
from .schemas import Concept, SchemaError

MAX_BODY_BYTES = 16 * 1024
MAX_TITLE = 120
MAX_DESCRIPTION = 2000
MAX_PANEL = 60
DEFAULT_PANEL, DEFAULT_SEED, MAX_SEED = 12, 42, 2**31 - 1

_STUDY_PATH = re.compile(r"/studies/([A-Za-z0-9-]{1,64})/(" + "|".join(re.escape(a) for a in ARTIFACTS) + r")")
_CONTENT_TYPES = {"html": "text/html; charset=utf-8", "json": "application/json; charset=utf-8"}
_CSP = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"

_PAGE_CSS = (
    "body{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:2em auto;padding:0 16px;color:#111;background:#fcfcfb}"
    ".banner{background:#fff3cd;color:#4a3500;border:2px solid #b07d00;border-radius:8px;padding:10px 14px;font-weight:700}"
    "label{display:block;margin-top:1em;font-weight:600}input,textarea{width:100%;font:inherit;padding:8px;box-sizing:border-box}"
    "button{margin-top:1.2em;font:inherit;padding:8px 18px}.err{color:#8a1c1c;font-weight:600}"
    "@media(prefers-color-scheme:dark){body{background:#1a1a19;color:#fff}.banner{background:#3a2e08;color:#ffe9a8}.err{color:#ff9b9b}}"
)


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_e(title)}</title><style>{_PAGE_CSS}</style></head><body>"
            f'<div class="banner" role="note">&#9888; {_e(LABEL)}</div>{body}</body></html>')


def render_form(values: dict[str, str] | None = None, error: str | None = None, csrf_token: str = "") -> str:
    v = {"concept_title": "", "concept_description": "", "panel_size": str(DEFAULT_PANEL), "seed": str(DEFAULT_SEED), **(values or {})}
    err = f'<p class="err" role="alert">{_e(error)}</p>' if error else ""
    return _page("Concept simulation", f"""<h1>Simulate a concept test</h1>
<p>Runs the offline, deterministic demo engine over a panel of synthetic personas. Nothing leaves your machine.</p>{err}
<form action="/run" method="post">
<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
<label for="t">Concept title</label><input id="t" name="concept_title" maxlength="{MAX_TITLE}" required value="{_e(v['concept_title'])}">
<label for="d">Concept description</label><textarea id="d" name="concept_description" rows="6" maxlength="{MAX_DESCRIPTION}" required>{_e(v['concept_description'])}</textarea>
<label for="n">Panel size (1-{MAX_PANEL})</label><input id="n" name="panel_size" inputmode="numeric" value="{_e(v['panel_size'])}">
<label for="s">Seed</label><input id="s" name="seed" inputmode="numeric" value="{_e(v['seed'])}">
<button type="submit">Run simulation</button></form>""")


def _int_field(fields: dict[str, list[str]], name: str, default: int, low: int, high: int) -> int:
    raw = (fields.get(name) or [""])[0].strip()
    if not raw:
        return default
    if not re.fullmatch(r"[0-9]{1,10}", raw) or not low <= int(raw) <= high:
        raise SchemaError(f"{name.replace('_', ' ')} must be an integer {low}-{high}")
    return int(raw)


def parse_run_request(fields: dict[str, list[str]]) -> tuple[Concept, int, int]:
    """Validate form fields (parse_qs shape). Raises SchemaError with a user-safe message."""
    first = lambda name: (fields.get(name) or [""])[0].strip()
    if first("engine") not in ("", "demo"):
        raise SchemaError("the web app only runs the demo engine; use the CLI for mlx")
    title, description = first("concept_title"), first("concept_description")
    if len(title) > MAX_TITLE:
        raise SchemaError(f"concept title must be at most {MAX_TITLE} characters")
    if len(description) > MAX_DESCRIPTION:
        raise SchemaError(f"concept description must be at most {MAX_DESCRIPTION} characters")
    concept = Concept(title, description)
    return concept, _int_field(fields, "panel_size", DEFAULT_PANEL, 1, MAX_PANEL), _int_field(fields, "seed", DEFAULT_SEED, 0, MAX_SEED)


class Handler(BaseHTTPRequestHandler):
    server_version = "ParrotBaseline/0.1"

    def log_message(self, format: str, *args: object) -> None:  # quiet by default
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def _send(self, status: int, body: str, kind: str = "html") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", _CONTENT_TYPES[kind])
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, values: dict[str, str] | None = None) -> None:
        self.close_connection = True
        self._send(status, render_form(values, message, self.server.csrf_token))

    def _valid_host(self) -> bool:
        host = self.headers.get("Host", "")
        try:
            hostname = urlsplit("//" + host).hostname
        except ValueError:
            return False
        return bool(hostname and hostname.lower() in self.server.allowed_hosts)

    def _valid_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin == f"http://{self.headers.get('Host', '')}"

    def do_GET(self) -> None:
        if not self._valid_host():
            return self._error(421, "Unrecognized Host header")
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(200, render_form(csrf_token=self.server.csrf_token))
        match = _STUDY_PATH.fullmatch(path)  # strict whitelist on the raw path: no unquoting, no "..", no slashes
        if match:
            study_dir = self.server.studies_dir / match.group(1)
            target = study_dir / match.group(2)
            try:
                root = self.server.studies_dir.resolve()
                resolved_study = study_dir.resolve()
                resolved_target = target.resolve(strict=True)
                safe = (not study_dir.is_symlink() and not target.is_symlink()
                        and resolved_study.parent == root and resolved_target.parent == resolved_study)
                if safe and resolved_target.is_file():
                    kind = "html" if target.suffix == ".html" else "json"
                    return self._send(200, resolved_target.read_text(encoding="utf-8"), kind)
            except (FileNotFoundError, NotADirectoryError):
                pass
            except (OSError, UnicodeError):
                return self._error(500, "Could not read study artifact")
        self._error(404, "Not found")

    def do_POST(self) -> None:
        if not self._valid_host():
            return self._error(421, "Unrecognized Host header")
        if not self._valid_origin():
            return self._error(403, "Cross-origin requests are not allowed")
        if self.path.split("?", 1)[0] != "/run":
            return self._error(404, "Not found")
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/x-www-form-urlencoded":
            return self._error(415, "Content-Type must be application/x-www-form-urlencoded")
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return self._error(411, "Content-Length required")
        if not re.fullmatch(r"[0-9]{1,12}", length_header.strip()):
            return self._error(400, "Invalid Content-Length")
        length = int(length_header)
        if length > MAX_BODY_BYTES:  # rejected before reading a single body byte
            return self._error(413, f"Request body too large (limit {MAX_BODY_BYTES} bytes)")
        try:
            fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True, strict_parsing=False)
        except UnicodeDecodeError:
            return self._error(400, "Body must be UTF-8")
        values = {k: v[0] for k, v in fields.items() if k in ("concept_title", "concept_description", "panel_size", "seed") and v}
        supplied_token = (fields.get("csrf_token") or [""])[0]
        if not secrets.compare_digest(supplied_token, self.server.csrf_token):
            return self._error(403, "Invalid or missing CSRF token", values)
        try:
            concept, panel_size, seed = parse_run_request(fields)
            now = datetime.now(timezone.utc)
            study_id = f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
            result = run_study(concept, DemoEngine(), self.server.studies_dir / study_id,
                               panel_size=panel_size, seed=seed, now=now, study_id=study_id)
        except SchemaError as exc:
            return self._error(400, str(exc), values)
        except (OSError, UnicodeError):
            return self._error(500, "Could not save study", values)
        self._send(200, _result_page(study_id, result.summary, concept))


def _result_page(study_id: str, summary: dict, concept: Concept) -> str:
    rows = "".join(
        f"<tr><td>{_e(seg)}</td>" + "".join(f"<td>{m:.2f}</td>" for m in means.values()) + "</tr>"
        for seg, means in summary["means"]["by_segment"].items()
    )
    heads = "".join(f"<th>{_e(q)}</th>" for q in next(iter(summary["means"]["by_segment"].values()), {}))
    links = " &middot; ".join(f'<a href="/studies/{_e(study_id)}/{name}">{name}</a>' for name in ARTIFACTS)
    return _page("Study complete", f"""<h1>{_e(concept.title)}</h1>
<p><a href="/studies/{_e(study_id)}/report.html">Open the full report</a></p>
<table><thead><tr><th>segment</th>{heads}</tr></thead><tbody>{rows}</tbody></table>
<p>Files: {links}</p><p><a href="/">Run another</a></p>""")


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ParrotHTTPServer(ThreadingHTTPServer):
    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(self.read_timeout)
        return request, client_address

    def process_request(self, request, client_address):
        if not self.request_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n"
                    b"Content-Length: 0\r\nCache-Control: no-store\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.request_slots.release()


def make_server(
    host: str,
    port: int,
    studies_dir: str | Path,
    allow_network: bool = False,
    read_timeout: float = 10.0,
    max_concurrent: int = 16,
) -> ParrotHTTPServer:
    if not allow_network and not _is_loopback_host(host):
        raise ValueError("refusing non-loopback host without --allow-network")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")
    server = ParrotHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.read_timeout = read_timeout
    server.request_slots = threading.BoundedSemaphore(max_concurrent)
    server.studies_dir = Path(studies_dir)
    server.csrf_token = secrets.token_urlsafe(32)
    server.allowed_hosts = {host.lower()}
    try:
        if ipaddress.ip_address(host).is_loopback:
            server.allowed_hosts.update({"localhost", "127.0.0.1", "::1"})
    except ValueError:
        pass
    server.verbose = False
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parrot_baseline.web", description="Local web UI for the demo engine.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--studies-dir", default=".artifacts/web")
    parser.add_argument("--allow-network", action="store_true", help="allow binding to a non-loopback host (no authentication)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = make_server(args.host, args.port, args.studies_dir, allow_network=args.allow_network)
    server.verbose = True
    print(f"Serving on http://{args.host}:{server.server_address[1]}/ (studies in {args.studies_dir}). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
