import os
import time
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


class HtmlCrawler:
    def __init__(
        self,
        start_url: str,
        output_dir: str = "downloaded_html",
        delay: float = 0.5,
        timeout: int = 15,
        max_pages: int = 200,
        user_agent: str = "Mozilla/5.0 (compatible; HtmlCrawler/1.0)"
    ):
        self.start_url = self._normalize_url(start_url)
        self.output_dir = output_dir
        self.delay = delay
        self.timeout = timeout
        self.max_pages = max_pages
        self.user_agent = user_agent

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        self.visited = set()
        self.found_html_pages = []

        parsed = urlparse(self.start_url)
        self.base_scheme = parsed.scheme
        self.base_netloc = parsed.netloc

        self.robot_parser = self._load_robots_txt()

        os.makedirs(self.output_dir, exist_ok=True)

    def _normalize_url(self, url: str) -> str:
        # Remove fragments (#section)
        url, _ = urldefrag(url)
        return url.strip()

    def _load_robots_txt(self):
        robots_url = f"{self.base_scheme}://{self.base_netloc}/robots.txt"
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            # If robots.txt can't be loaded, you can choose to allow or deny.
            # Here we default to allow.
            pass
        return rp

    def _allowed_by_robots(self, url: str) -> bool:
        try:
            return self.robot_parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _is_same_domain(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.netloc == self.base_netloc

    def _is_html_response(self, response: requests.Response) -> bool:
        content_type = response.headers.get("Content-Type", "").lower()
        return "text/html" in content_type or "application/xhtml+xml" in content_type

    def _safe_filename(self, url: str) -> str:
        """
        Convert URL into a safe local file path.
        Example:
            https://example.com/a/b -> downloaded_html/a/b.html
            https://example.com/     -> downloaded_html/index.html
        """
        parsed = urlparse(url)
        path = parsed.path

        if not path or path.endswith("/"):
            path = path + "index.html"

        # If the path has no extension, assume HTML
        if "." not in os.path.basename(path):
            path = path.rstrip("/") + ".html"

        # Remove leading slash
        path = path.lstrip("/")

        # If query string exists, append a short hash to avoid collisions
        if parsed.query:
            name, ext = os.path.splitext(path)
            query_hash = hashlib.md5(parsed.query.encode("utf-8")).hexdigest()[:8]
            path = f"{name}_{query_hash}{ext or '.html'}"

        local_path = os.path.join(self.output_dir, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        return local_path

    def _save_html(self, url: str, html: str):
        file_path = self._safe_filename(url)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[SAVED] {url} -> {file_path}")

    def _extract_links(self, current_url: str, html: str):
        soup = BeautifulSoup(html, "html.parser")
        links = set()

        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href:
                continue

            # Resolve relative URLs
            absolute_url = urljoin(current_url, href)
            absolute_url = self._normalize_url(absolute_url)

            # Only same domain
            if not self._is_same_domain(absolute_url):
                continue

            # Skip non-http(s)
            parsed = urlparse(absolute_url)
            if parsed.scheme not in ("http", "https"):
                continue

            links.add(absolute_url)

        return links

    def crawl(self):
        queue = deque([self.start_url])

        while queue and len(self.visited) < self.max_pages:
            url = queue.popleft()

            if url in self.visited:
                continue

            if not self._allowed_by_robots(url):
                print(f"[SKIP robots.txt] {url}")
                continue

            try:
                print(f"[FETCH] {url}")
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                final_url = self._normalize_url(response.url)

                if final_url in self.visited:
                    continue

                self.visited.add(final_url)

                if response.status_code != 200:
                    print(f"[SKIP status={response.status_code}] {final_url}")
                    continue

                if not self._is_html_response(response):
                    print(f"[SKIP non-HTML] {final_url}")
                    continue

                html = response.text
                self.found_html_pages.append(final_url)
                self._save_html(final_url, html)

                # Extract and enqueue more same-domain links
                for link in self._extract_links(final_url, html):
                    if link not in self.visited:
                        queue.append(link)

                time.sleep(self.delay)

            except requests.RequestException as e:
                print(f"[ERROR] {url}: {e}")

        return self.found_html_pages


if __name__ == "__main__":
    START_URL = "https://example.com"
    crawler = HtmlCrawler(
        start_url=START_URL,
        output_dir="downloaded_html",
        delay=0.5,       # be polite
        timeout=15,
        max_pages=100    # prevent runaway crawling
    )

    pages = crawler.crawl()

    print("\n=== Crawl finished ===")
    print(f"Total HTML pages found: {len(pages)}")
    for p in pages:
        print(p)
