import os
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"
OUT_DIR = "bridge_dataset_1.0.0"
MAX_WORKERS = 8

os.makedirs(OUT_DIR, exist_ok=True)


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


def collect_files(url):
    files = []
    stack = [url]
    seen = set()

    while stack:
        current = stack.pop()

        if current in seen:
            continue
        seen.add(current)

        for link in get_links(current):
            if link.endswith("/"):
                stack.append(link)
            else:
                files.append(link)

    return files


def get_remote_size(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        return int(r.headers.get("content-length", 0))
    except Exception:
        return 0


def download_file(url, pbar):
    relative_path = url[len(BASE_URL):]
    out_path = os.path.join(OUT_DIR, relative_path)
    tmp_path = out_path + ".part"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    remote_size = get_remote_size(url)

    if os.path.exists(out_path):
        local_size = os.path.getsize(out_path)

        if remote_size == 0 or local_size == remote_size:
            pbar.update(remote_size if remote_size else local_size)
            return f"Skipped: {relative_path}"

    downloaded = 0

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))

    os.replace(tmp_path, out_path)
    return f"Downloaded: {relative_path}"


def main():
    print("Collecting file list...")
    files = collect_files(BASE_URL)

    print(f"Found {len(files)} files.")

    sizes = [get_remote_size(url) for url in files]
    total_size = sum(sizes)

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(download_file, url, pbar) for url in files]

            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print(f"Error: {e}")

    print("Done.")


if __name__ == "__main__":
    main()