import io
import subprocess
import sys
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from download_speed import (
    BYTES_PER_MIB,
    CHUNK_SIZE,
    BenchmarkResult,
    DownloadError,
    Measurement,
    main,
    run_benchmark,
)

PAYLOAD = bytes(range(256)) * 600
BODY_DELAY_SECONDS = 0.02
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "download_speed.py"


class TestServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), TestHandler)
        self.events: list[tuple[str, str]] = []
        self.events_lock = threading.Lock()

    def record(self, event: str, path: str) -> None:
        with self.events_lock:
            self.events.append((event, path))


class TestHandler(BaseHTTPRequestHandler):
    server: TestServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.server.record("start", self.path)
        if self.path in {
            "/redirect", "/large-redirect", "/truncated-redirect", "/loop"
        }:
            body = b"" if self.path == "/redirect" else PAYLOAD
            target = "/loop" if self.path == "/loop" else "/body"
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.path == "/truncated-redirect":
                body = body[:100]
            self.wfile.write(body)
            return
        if self.path == "/error":
            self.send_error(404)
            return
        if self.path == "/empty":
            self.send_response(204)
            self.end_headers()
            return

        self.send_response(200)
        if self.path in {"/chunked", "/broken-chunked"}:
            self.send_header("Transfer-Encoding", "chunked")
        elif self.path != "/no-length":
            self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()

        if self.path == "/timeout":
            time.sleep(0.15)
            return
        if self.path == "/truncated":
            self.wfile.write(PAYLOAD[:100])
            return
        if self.path == "/broken-chunked":
            self.wfile.write(b"100\r\nshort")
            return
        if self.path == "/chunked":
            for start in range(0, len(PAYLOAD), 8192):
                chunk = PAYLOAD[start:start + 8192]
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        if self.path == "/delayed":
            self.wfile.write(PAYLOAD[:100])
            self.wfile.flush()
            time.sleep(BODY_DELAY_SECONDS)
            self.server.record("body", self.path)
            self.wfile.write(PAYLOAD[100:])
            return
        self.wfile.write(PAYLOAD)


class StatisticsTests(unittest.TestCase):
    def test_aggregate_speed_uses_total_bytes_and_total_time(self) -> None:
        result = BenchmarkResult((
            Measurement(BYTES_PER_MIB, 1.0),
            Measurement(9 * BYTES_PER_MIB, 3.0),
        ))

        self.assertEqual(result.total_bytes, 10 * BYTES_PER_MIB)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.average_seconds, 2.0)
        self.assertEqual(result.speed_mib_per_second, 2.5)

    def test_failure_stops_series_without_retries(self) -> None:
        with patch(
            "download_speed.measure_download",
            side_effect=[Measurement(10, 1.0), OSError("connection lost")],
        ) as download:
            with self.assertRaisesRegex(DownloadError, "запрос 2/10"):
                run_benchmark("https://example.com/file")

        self.assertEqual(download.call_count, 2)


class HttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy_patch = patch.dict(
            "os.environ", {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}
        )
        self.proxy_patch.start()
        self.addCleanup(self.proxy_patch.stop)
        self.server = TestServer()
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_ten_sequential_downloads_wait_for_the_complete_body(self) -> None:
        result = run_benchmark(self.base_url + "/delayed")

        self.assertEqual(len(result.measurements), 10)
        self.assertEqual(result.total_bytes, 10 * len(PAYLOAD))
        self.assertEqual(
            self.server.events,
            [("start", "/delayed"), ("body", "/delayed")] * 10,
        )
        for measurement in result.measurements:
            self.assertGreaterEqual(
                measurement.elapsed_seconds, BODY_DELAY_SECONDS
            )
            self.assertEqual(measurement.bytes_downloaded, len(PAYLOAD))

    def test_chunked_body_counts_payload_bytes(self) -> None:
        result = run_benchmark(self.base_url + "/chunked")

        self.assertEqual(result.total_bytes, 10 * len(PAYLOAD))
        self.assertEqual(len(self.server.events), 10)

    def test_body_without_content_length_reads_until_eof(self) -> None:
        result = run_benchmark(self.base_url + "/no-length")

        self.assertEqual(result.total_bytes, 10 * len(PAYLOAD))

    def test_redirects_are_included_in_each_download(self) -> None:
        result = run_benchmark(self.base_url + "/redirect")

        self.assertEqual(result.total_bytes, 10 * len(PAYLOAD))
        self.assertEqual(
            self.server.events,
            [("start", "/redirect"), ("start", "/body")] * 10,
        )

    def test_empty_response_has_zero_speed(self) -> None:
        result = run_benchmark(self.base_url + "/empty")

        self.assertEqual(len(result.measurements), 10)
        self.assertEqual(result.total_bytes, 0)
        self.assertEqual(result.speed_mib_per_second, 0.0)

    def test_large_redirect_body_is_read_in_bounded_chunks(self) -> None:
        original_read = HTTPResponse.read
        read_sizes = []
        redirect_responses = []

        def checked_read(
            response: HTTPResponse, amount: int | None = None
        ) -> bytes:
            if response.status == 302 and response.fp is not None:
                self.assertIsNotNone(amount)
                self.assertGreater(amount, 0)
                self.assertLessEqual(amount, CHUNK_SIZE)
                read_sizes.append(amount)
                if response not in redirect_responses:
                    redirect_responses.append(response)
            return original_read(response, amount)

        with patch.object(HTTPResponse, "read", new=checked_read):
            result = run_benchmark(self.base_url + "/large-redirect")

        self.assertGreater(len(read_sizes), 10)
        self.assertEqual(len(redirect_responses), 10)
        self.assertTrue(
            all(response.closed for response in redirect_responses)
        )
        self.assertEqual(result.total_bytes, 10 * len(PAYLOAD))
        self.assertEqual(
            self.server.events,
            [("start", "/large-redirect"), ("start", "/body")] * 10,
        )

    def test_truncated_redirect_body_stops_before_following_location(
        self,
    ) -> None:
        with self.assertRaisesRegex(DownloadError, "неполный ответ"):
            run_benchmark(self.base_url + "/truncated-redirect")

        self.assertEqual(
            self.server.events, [("start", "/truncated-redirect")]
        )

    def test_redirect_loop_remains_bounded(self) -> None:
        with self.assertRaisesRegex(DownloadError, "запрос 1/10: HTTP 302"):
            run_benchmark(self.base_url + "/loop")

        self.assertEqual(self.server.events, [("start", "/loop")] * 5)

    def test_truncated_content_length_is_an_error(self) -> None:
        with self.assertRaisesRegex(DownloadError, "неполный ответ"):
            run_benchmark(self.base_url + "/truncated")

        self.assertEqual(len(self.server.events), 1)

    def test_truncated_chunked_body_is_an_error(self) -> None:
        with self.assertRaisesRegex(DownloadError, "запрос 1/10"):
            run_benchmark(self.base_url + "/broken-chunked")

        self.assertEqual(len(self.server.events), 1)

    def test_http_error_stops_immediately(self) -> None:
        with self.assertRaisesRegex(DownloadError, "запрос 1/10: HTTP 404"):
            run_benchmark(self.base_url + "/error")

        self.assertEqual(len(self.server.events), 1)

    def test_read_timeout_stops_immediately(self) -> None:
        with self.assertRaisesRegex(DownloadError, "запрос 1/10"):
            run_benchmark(self.base_url + "/timeout", timeout=0.03)

        self.assertEqual(len(self.server.events), 1)

    def test_cli_process_outputs_measured_statistics(self) -> None:
        process = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), self.base_url + "/body"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertIn("Выполнено запросов: 10", process.stdout)
        self.assertIn(
            f"Всего скачано: {10 * len(PAYLOAD)} байт", process.stdout
        )
        self.assertIn("Среднее время запроса:", process.stdout)
        self.assertIn("MiB/s", process.stdout)
        self.assertEqual(len(self.server.events), 10)


class CliTests(unittest.TestCase):
    def test_invalid_arguments_fail_before_network_access(self) -> None:
        invalid_arguments = [
            [],
            ["ftp://example.com/file"],
            ["file:///etc/hosts"],
            ["example.com"],
            ["https://"],
            ["http://[invalid"],
            ["http://example.com:70000"],
            ["http://example.com:0"],
            ["http://example.com/a b"],
            ["https://example.com", "--timeout", "0"],
            ["https://example.com", "--timeout", "-1"],
            ["https://example.com", "--timeout", "nan"],
            ["https://example.com", "--timeout", "inf"],
            ["https://example.com", "--timeout", "invalid"],
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with patch("download_speed.build_opener") as make_opener:
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main(arguments)
                    self.assertEqual(raised.exception.code, 2)
                    make_opener.assert_not_called()

    def test_failure_prints_only_error_and_returns_one(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch(
            "download_speed.run_benchmark", side_effect=DownloadError("сбой")
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(["https://example.com"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Ошибка: сбой", stderr.getvalue())

    def test_keyboard_interrupt_returns_130(self) -> None:
        with patch(
            "download_speed.run_benchmark", side_effect=KeyboardInterrupt
        ):
            with redirect_stderr(io.StringIO()):
                exit_code = main(["https://example.com"])

        self.assertEqual(exit_code, 130)


if __name__ == "__main__":
    unittest.main()
