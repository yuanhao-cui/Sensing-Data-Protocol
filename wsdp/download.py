import os
import sys
import getpass
import requests
import kagglehub

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import load_api, load_mapping, download_ftp


def download(dataset_name: str, dest: str):
    dn = load_mapping(dataset_name)
    if dataset_name != 'elderAL':
        try:
            _download_without_aws(dataset_name, dest)
            return
        except Exception as e:
            print(f"Error occurred when tried to download with other sources: {e}, try to download with SDP Storage\n")

    email = input("Email: ").strip()
    password = getpass.getpass("Password: ")

    if not email or not password:
        print("error: email and password cannot be null")
        return

    api = load_api("auth")
    print(f"prepare to download: {dataset_name}")

    payload = {
        "email": email,
        "password": password,
        "fileKey": dn
    }

    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "userEmail": email
        }
        response = requests.post(api, headers=headers, json=payload, timeout=20)

        if response.status_code == 200:
            url = response.text
            _download_file_from_url(url, dest, dn)

        elif response.status_code == 401:
            print("Authentication failed, incorrect email or password")
        elif response.status_code == 404:
            print("Specified dataset does not exist")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")


def _download_file_from_url(url, dest, file_name):
    chunk_size = 16 * 1024 * 1024
    max_workers = 8

    if not os.path.exists(dest):
        os.makedirs(dest)
    local_path = os.path.join(dest, file_name)

    try:
        head_resp = requests.head(url)
        file_size = int(head_resp.headers.get('content-length', 0))
    except Exception as e:
        print("cannot get size of dataset，download with single thread...")
        file_size = 0

    if file_size <= 0:
        _single_thread_download(url, local_path, file_name)
        return

    size_str = f"{file_size/ (1024 ** 2):.2f} MB" if file_size < 1024 ** 3 else f"{file_size / (1024 ** 3):.2f} GB"
    print("file_size: " + size_str)

    with open(local_path, "wb") as f:
        f.seek(file_size - 1)
        f.write(b'\0')

    chunks = []
    for start in range(0, file_size, chunk_size):
        end = min(start + chunk_size - 1, file_size - 1)
        chunks.append((start, end))

    def download_chunk(chunk_range):
        st, ed = chunk_range
        headers = {'Range': f'bytes={st}-{ed}'}

        # retry
        for _ in range(3):
            try:
                r = requests.get(url, headers=headers, stream=True, timeout=20)
                r.raise_for_status()

                with open(local_path, "r+b") as f:
                    f.seek(st)
                    f.write(r.content)
                return len(r.content)
            except Exception:
                continue
        raise Exception(f"download chunk failed: {st}-{ed}")

    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=file_name) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(download_chunk, chunk): chunk for chunk in chunks}

            for future in as_completed(future_to_chunk):
                try:
                    bytes_written = future.result()
                    pbar.update(bytes_written)
                except Exception as exc:
                    print(f'\nfailed to download a particular chunk，stop downloading: {exc}')
                    # TODO Resumable
                    sys.exit(1)

    # TODO Compare Hash of downloaded file with Hash return from backend


def _single_thread_download(url, dest, file_name):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()

            total_size = int(r.headers.get('content-length', 0))

            block_size = 16384  # 16KB

            # if total_size is 0，tqdm switch display model automatically
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=file_name) as pbar:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        print("\ndownload finished with single thread")
    except Exception as e:
        print(f"\ndownload failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)


def _download_without_aws(dataset_name: str, dest: str):
    if dataset_name == 'widar':
        download_ftp(dataset_name, dest)
    elif dataset_name == 'gait':
        download_ftp(dataset_name, dest)
    elif dataset_name == 'xrf55':
        os.environ['KAGGLEHUB_CACHE'] = dest
        print(f"os.environ['KAGGLEHUB_CACHE'] is changed to {dest}")
        path = kagglehub.dataset_download("xrfdataset/xrf55-rawdata")
        print("Path to dataset files:", path)
