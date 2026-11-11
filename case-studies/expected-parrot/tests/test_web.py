import http.client
import json
import re
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlencode

from parrot_baseline import web
from parrot_baseline.schemas import SchemaError

FORM = {"concept_title": "Spicy Tea", "concept_description": "Chili-infused iced tea with honey."}
XSS = '<script>alert("x")</script>'


class ParseRunRequestTests(unittest.TestCase):
    def test_defaults(self):
        concept, size, seed = web.parse_run_request({k: [v] for k, v in FORM.items()})
        self.assertEqual((concept.title, size, seed), ("Spicy Tea", 12, 42))

    def test_rejects_bad_input(self):
        ok = {k: [v] for k, v in FORM.items()}
        bad_cases = [
            {**ok, "concept_title": [""]},
            {**ok, "concept_title": ["x" * (web.MAX_TITLE + 1)]},
            {**ok, "concept_description": ["x" * (web.MAX_DESCRIPTION + 1)]},
            {**ok, "panel_size": ["abc"]},
            {**ok, "panel_size": ["0"]},
            {**ok, "panel_size": [str(web.MAX_PANEL + 1)]},
            {**ok, "seed": ["-1"]},
            {**ok, "seed": ["1.5"]},
            {**ok, "engine": ["mlx"]},
            {"concept_title": ["x"]},
        ]
        for case in bad_cases:
            with self.assertRaises(SchemaError, msg=str(case)):
                web.parse_run_request(case)


class WebServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        (cls.root / "secret.txt").write_text("TOP SECRET")
        cls.server = web.make_server("127.0.0.1", 0, cls.root / "studies")
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)
        cls._tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", "replace")
        conn.close()
        return resp, data

    def post_form(self, fields):
        fields = {**fields, "csrf_token": self.server.csrf_token}
        return self.request("POST", "/run", urlencode(fields), {"Content-Type": "application/x-www-form-urlencoded"})

    def raw(self, payload: bytes) -> str:
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as s:
            s.sendall(payload)
            chunks = []
            while True:
                try:
                    chunk = s.recv(65536)
                except ConnectionResetError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", "replace")

    def test_get_root_renders_form(self):
        resp, body = self.request("GET", "/")
        self.assertEqual(resp.status, 200)
        self.assertIn("text/html", resp.getheader("Content-Type"))
        self.assertIn('action="/run"', body)
        self.assertIn('method="post"', body.lower())
        for name in ("concept_title", "concept_description", "panel_size", "seed"):
            self.assertIn(f'name="{name}"', body)
        self.assertIn("not human evidence", body.lower())

    def test_post_requires_per_server_csrf_token(self):
        _, form = self.request("GET", "/")
        match = re.search(r'name="csrf_token" value="([^"]+)"', form)
        self.assertIsNotNone(match, form)
        self.assertEqual(match.group(1), self.server.csrf_token)
        resp, _ = self.request("POST", "/run", urlencode(FORM), {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(resp.status, 403)

    def test_rejects_unrecognized_host(self):
        fields = {**FORM, "csrf_token": self.server.csrf_token}
        resp, _ = self.request("POST", "/run", urlencode(fields), {
            "Content-Type": "application/x-www-form-urlencoded", "Host": "evil.example",
        })
        self.assertEqual(resp.status, 421)

    def test_rejects_cross_origin_post(self):
        fields = {**FORM, "csrf_token": self.server.csrf_token}
        resp, _ = self.request("POST", "/run", urlencode(fields), {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://evil.example",
        })
        self.assertEqual(resp.status, 403)

    def test_post_run_persists_and_links_report(self):
        resp, body = self.post_form(FORM)
        self.assertEqual(resp.status, 200)
        match = re.search(r'href="(/studies/([A-Za-z0-9-]+)/report\.html)"', body)
        self.assertIsNotNone(match, body)
        study_dir = self.root / "studies" / match.group(2)
        for name in ("manifest.json", "responses.json", "summary.json", "report.html"):
            self.assertTrue((study_dir / name).is_file(), name)
        report_resp, report = self.request("GET", match.group(1))
        self.assertEqual(report_resp.status, 200)
        self.assertIn("text/html", report_resp.getheader("Content-Type"))
        self.assertIn("Simulated hypotheses, not human evidence", report)
        self.assertIn("Spicy Tea", report)
        manifest_resp, manifest = self.request("GET", f"/studies/{match.group(2)}/manifest.json")
        self.assertEqual(json.loads(manifest)["engine"], "demo")
        self.assertIn("application/json", manifest_resp.getheader("Content-Type"))

    def test_filesystem_error_returns_controlled_500(self):
        with mock.patch.object(web, "run_study", side_effect=OSError("disk unavailable")):
            resp, body = self.post_form(FORM)
        self.assertEqual(resp.status, 500)
        self.assertIn("could not save study", body.lower())
        self.assertNotIn("Traceback", body)

    def test_user_input_escaped_in_result_report_and_error_pages(self):
        resp, body = self.post_form({"concept_title": XSS, "concept_description": XSS})
        self.assertEqual(resp.status, 200)
        self.assertNotIn("<script", body.lower())
        link = re.search(r'href="(/studies/[^"]+/report\.html)"', body).group(1)
        self.assertNotIn("<script", self.request("GET", link)[1].lower())
        resp, body = self.post_form({"concept_title": "", "concept_description": XSS, "panel_size": XSS})
        self.assertEqual(resp.status, 400)
        self.assertNotIn("<script", body.lower())
        self.assertIn("&lt;script&gt;", body)

    def test_security_headers(self):
        resp, _ = self.request("GET", "/")
        self.assertEqual(resp.getheader("X-Content-Type-Options"), "nosniff")
        self.assertIn("default-src 'none'", resp.getheader("Content-Security-Policy"))

    def test_invalid_fields_rejected_with_400(self):
        for fields in ({**FORM, "panel_size": "abc"}, {**FORM, "panel_size": "9999"}, {**FORM, "engine": "mlx"},
                       {"concept_title": "", "concept_description": "x"}):
            self.assertEqual(self.post_form(fields)[0].status, 400, fields)

    def test_oversized_body_rejected_413(self):
        big = urlencode({**FORM, "concept_description": "x" * (web.MAX_BODY_BYTES + 10)})
        self.assertEqual(self.post_form_raw(big), 413)

    def post_form_raw(self, body):
        head = ("POST /run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/x-www-form-urlencoded\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n")
        return int(self.raw(head.encode() + body[:100].encode()).split()[1])

    def test_declared_huge_content_length_rejected_without_waiting(self):
        reply = self.raw(b"POST /run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/x-www-form-urlencoded\r\n"
                         b"Content-Length: 99999999\r\nConnection: close\r\n\r\nabc")
        self.assertEqual(int(reply.split()[1]), 413)

    def test_missing_or_invalid_content_length(self):
        base = "POST /run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/x-www-form-urlencoded\r\nConnection: close\r\n"
        self.assertEqual(int(self.raw((base + "\r\n").encode()).split()[1]), 411)
        self.assertEqual(int(self.raw((base + "Content-Length: abc\r\n\r\n").encode()).split()[1]), 400)
        self.assertEqual(int(self.raw((base + "Content-Length: -5\r\n\r\n").encode()).split()[1]), 400)

    def test_wrong_content_type_rejected(self):
        resp, _ = self.request("POST", "/run", json.dumps(FORM), {"Content-Type": "application/json"})
        self.assertEqual(resp.status, 415)

    def test_non_utf8_body_rejected(self):
        resp, _ = self.request("POST", "/run", b"concept_title=%ff\xff", {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(resp.status, 400)

    def test_symlink_artifact_is_not_served(self):
        study = self.root / "studies" / "linked"
        study.mkdir(parents=True, exist_ok=True)
        (study / "report.html").symlink_to(self.root / "secret.txt")
        resp, body = self.request("GET", "/studies/linked/report.html")
        self.assertEqual(resp.status, 404)
        self.assertNotIn("TOP SECRET", body)

    def test_path_traversal_blocked(self):
        _, body = self.post_form(FORM)
        study_id = re.search(r"/studies/([A-Za-z0-9-]+)/report", body).group(1)
        paths = [
            "/studies/../secret.txt", "/studies/..%2fsecret.txt", "/studies/%2e%2e/secret.txt",
            f"/studies/{study_id}/../../secret.txt", f"/studies/{study_id}/..%2f..%2fsecret.txt",
            f"/studies/{study_id}/secret.txt", f"/studies/{study_id}/", "/studies/", "/studies/nope/report.html",
            "/studies/....//report.html", "//etc/passwd", "/../secret.txt",
        ]
        for path in paths:
            resp, data = self.request("GET", path)
            self.assertEqual(resp.status, 404, path)
            self.assertNotIn("TOP SECRET", data, path)

    def test_unknown_routes(self):
        self.assertEqual(self.request("GET", "/nope")[0].status, 404)
        self.assertEqual(self.request("POST", "/elsewhere", "", {"Content-Length": "0"})[0].status, 404)

    def test_incomplete_request_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = web.make_server("127.0.0.1", 0, tmp, read_timeout=0.1)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with socket.create_connection(server.server_address, timeout=2) as sock:
                    sock.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1")
                    sock.settimeout(2)
                    self.assertEqual(sock.recv(1024), b"")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_concurrency_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = web.make_server("127.0.0.1", 0, tmp, read_timeout=2, max_concurrent=1)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            blocker = socket.create_connection(server.server_address, timeout=2)
            try:
                blocker.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1")
                with socket.create_connection(server.server_address, timeout=2) as second:
                    second.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                    second.settimeout(2)
                    reply = second.recv(1024)
                self.assertIn(b" 503 ", reply)
            finally:
                blocker.close()
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_network_bind_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "allow-network"):
                web.make_server("0.0.0.0", 0, tmp)
        args = web.build_parser().parse_args(["--host", "0.0.0.0", "--allow-network"])
        self.assertTrue(args.allow_network)

    def test_main_parser_defaults(self):
        args = web.build_parser().parse_args(["--port", "8765"])
        self.assertEqual((args.port, args.host), (8765, "127.0.0.1"))


if __name__ == "__main__":
    unittest.main()
