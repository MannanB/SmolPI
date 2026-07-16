import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"
OUT_DIR = "bridge_dataset_1.0.0"
MAX_WORKERS = 8
CHUNK_SIZE = 1024 * 1024 * 4  # 4 MB

os.makedirs(OUT_DIR, exist_ok=True)

thread_local = threading.local()
worker_ids = Queue()

for i in range(MAX_WORKERS):
    worker_ids.put(i)


def init_worker():
    thread_local.worker_id = worker_ids.get()


def get_links(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    links = []

    for a in soup.find_all("a"):
        href = a.get("href")

        if not href:
            continue
        if href.startswith("?") or href.startswith("#") or href == "../":
            continue

        full_url = urljoin(url, href)

        if full_url.startswith(BASE_URL):
            links.append(full_url)

    return links


def collect_files():
    files = []
    stack = [BASE_URL]
    seen = set()

    while stack:
        url = stack.pop()

        if url in seen:
            continue

        seen.add(url)

        for link in get_links(url):
            if link.endswith("/"):
                stack.append(link)
            else:
                files.append(link)

    return files


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def download_file(url):
    session = get_session()

    rel_path = url[len(BASE_URL) :]
    out_path = os.path.join(OUT_DIR, rel_path)
    tmp_path = out_path + ".part"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        return f"Skipped existing: {rel_path}"

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        position = thread_local.worker_id + 1

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=rel_path[-40:],
            position=position,
            leave=False,
        ) as pbar:
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    os.replace(tmp_path, out_path)
    return f"Downloaded: {rel_path}"


def main():
    print("Collecting file URLs...")
    files = collect_files()
    print(f"Found {len(files)} files.")

    with tqdm(total=len(files), desc="Files", position=0) as file_pbar:
        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker,
        ) as executor:
            futures = [executor.submit(download_file, url) for url in files]

            for future in as_completed(futures):
                try:
                    msg = future.result()
                    tqdm.write(msg)
                except Exception as e:
                    tqdm.write(f"Error: {e}")

                file_pbar.update(1)

    print("Done.")


if __name__ == "__main__":
    main()
