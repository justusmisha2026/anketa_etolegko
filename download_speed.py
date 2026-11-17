import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from http.client import HTTPException, HTTPMessage, HTTPResponse
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REQUEST_COUNT = 10
CHUNK_SIZE = 64 * 1024
BYTES_PER_MIB = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0


class DownloadError(Exception):
    pass


@dataclass(frozen=True)
class Measurement:
    bytes_downloaded: int
    elapsed_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    measurements: tuple[Measurement, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes_downloaded for item in self.measurements)

    @property
    def total_seconds(self) -> float:
        return sum(item.elapsed_seconds for item in self.measurements)

    @property
    def average_seconds(self) -> float:
        return self.total_seconds / len(self.measurements)

    @property
    def speed_mib_per_second(self) -> float:
        return self.total_bytes / self.total_seconds / BYTES_PER_MIB


def _http_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("укажите HTTP(S)-адрес с именем хоста")
        if parsed.port == 0:
            raise ValueError("порт должен быть от 1 до 65535")
        if any(
            character.isspace() or ord(character) < 32
            for character in value
        ):
            raise ValueError(
                "URL не должен содержать пробелы и управляющие символы"
            )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "тайм-аут должен быть числом"
        ) from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError(
            "тайм-аут должен быть конечным числом > 0"
        )
    return timeout


def _read_body(response: HTTPResponse) -> int:
    expected_size = None
    content_length = response.getheader("Content-Length")
    transfer_encoding = response.getheader("Transfer-Encoding", "").lower()
    if (
        content_length is not None
        and transfer_encoding != "chunked"
        and response.status != 204
    ):
        expected_size = int(content_length)
        if expected_size < 0:
            raise DownloadError("сервер вернул отрицательный Content-Length")

    total_bytes = 0
    while chunk := response.read(CHUNK_SIZE):
        total_bytes += len(chunk)

    if expected_size is not None and total_bytes != expected_size:
        raise DownloadError(
            f"неполный ответ: получено {total_bytes} из {expected_size} байт"
        )
    return total_bytes


class StreamingRedirectHandler(HTTPRedirectHandler):
    """Read redirect bodies in chunks before urllib's unbounded read."""

    def redirect_request(
        self,
        req: Request,
        fp: HTTPResponse,
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        redirected_request = super().redirect_request(
            req, fp, code, msg, headers, newurl
        )
        if redirected_request is not None:
            with fp:
                _read_body(fp)
        return redirected_request


def measure_download(url: str, timeout: float) -> Measurement:
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "download-speed/1.0",
        },
    )
    opener = build_opener(StreamingRedirectHandler())
    started_at = perf_counter()
    with opener.open(request, timeout=timeout) as response:
        if response.status is None or not 200 <= response.status < 300:
            raise DownloadError(f"неуспешный HTTP-статус: {response.status}")
        total_bytes = _read_body(response)
    elapsed_seconds = perf_counter() - started_at
    if elapsed_seconds <= 0:
        raise DownloadError("не удалось измерить положительную длительность")
    return Measurement(total_bytes, elapsed_seconds)


def run_benchmark(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> BenchmarkResult:
    measurements = []
    for request_number in range(1, REQUEST_COUNT + 1):
        try:
            measurements.append(measure_download(url, timeout))
        except HTTPError as error:
            error.close()
            raise DownloadError(
                f"запрос {request_number}/{REQUEST_COUNT}: HTTP {error.code}"
            ) from error
        except (
            DownloadError,
            URLError,
            HTTPException,
            OSError,
            ValueError,
            OverflowError,
        ) as error:
            raise DownloadError(
                f"запрос {request_number}/{REQUEST_COUNT}: {error}"
            ) from error
    return BenchmarkResult(tuple(measurements))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Измерить скорость 10 последовательных HTTP(S)-загрузок.",
    )
    parser.add_argument(
        "url", type=_http_url, help="адрес скачиваемого ресурса"
    )
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "тайм-аут сетевой операции в секундах "
            "(по умолчанию: %(default)s)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        result = run_benchmark(args.url, args.timeout)
    except DownloadError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Измерение прервано пользователем.", file=sys.stderr)
        return 130

    print(f"Выполнено запросов: {len(result.measurements)}")
    print(f"Среднее время запроса: {result.average_seconds:.6f} с")
    print(
        f"Всего скачано: {result.total_bytes} байт "
        f"({result.total_bytes / BYTES_PER_MIB:.6f} MiB)"
    )
    print(f"Скорость скачивания: {result.speed_mib_per_second:.6f} MiB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
