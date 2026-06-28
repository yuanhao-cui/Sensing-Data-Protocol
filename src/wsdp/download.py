from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import os
import sys

import kagglehub
import requests
import urllib3
from tqdm import tqdm

from .utils import load_api, load_mapping, download_ftp

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download(dataset_name: str, dest: str, email: str = None, password: str = None, token: str = None, extensions: list = None):
    """
    Download a dataset by name.

    Supports three authentication methods (in order of preference):
    1. JWT Token (via `token` argument or WSDP_TOKEN env var)
    2. Email/password (via arguments or interactive prompt)
    3. Direct download for datasets that don't require auth

    Args:
        dataset_name: Name of the dataset to download
        dest: Destination directory
        email: Email for authentication (optional, for non-interactive mode)
        password: Password for authentication (optional, for non-interactive mode)
        token: JWT token for authentication (optional, overrides email/password)
        extensions: Optional list of file extensions to download (e.g. ['.csv', '.mat']).
                    Only FTP datasets support this filter.
    """
    dn = load_mapping(dataset_name)
    if dataset_name != 'elderAL':
        try:
            _download_without_aws(dataset_name, dest, extensions=extensions)
            return
        except Exception as e:
            print(f"Error occurred when tried to download with other sources: {e}, try to download with SDP Storage\n")

    # Determine authentication method
    auth_token = token or os.environ.get("WSDP_TOKEN")

    if auth_token:
        # JWT token auth
        print("Using JWT token authentication")
        api = load_api("auth")
        payload = {"fileKey": dn}
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Authorization": f"Bearer {auth_token}",
        }
    else:
        # Email/password auth — only prompt interactively if stdin is a TTY
        if not email:
            if sys.stdin.isatty():
                email = input("Email: ").strip()
            else:
                print("error: email is required (non-interactive mode). Use --email or set WSDP_TOKEN.")
                return
        if not password:
            if sys.stdin.isatty():
                password = getpass.getpass("Password: ")
            else:
                print("error: password is required (non-interactive mode). Use --password or set WSDP_TOKEN.")
                return

        if not email or not password:
            print("error: email and password cannot be empty")
            return

        api = load_api("auth")
        payload = {
            "email": email,
            "password": password,
            "fileKey": dn
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "userEmail": email
        }

    print(f"prepare to download: {dataset_name}")

    try:
        response = requests.post(api, headers=headers, json=payload, timeout=20, allow_redirects=True, verify=False)

        if response.status_code == 200:
            url = response.text
            _download_file_from_url(url, dest, dn)
        elif response.status_code == 401:
            print("Authentication failed, incorrect credentials or token")
        elif response.status_code == 404:
            print("Specified dataset does not exist")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")


def _download_file_from_url(url, dest, file_name):
    """Download a file from a backend-generated URL with multi-threaded chunked download."""

    chunk_size = 16 * 1024 * 1024
    max_workers = 8

    if not os.path.exists(dest):
        os.makedirs(dest)
    local_path = os.path.join(dest, file_name)

    try:
        head_resp = requests.head(url)
        if head_resp.status_code == 301 or head_resp.status_code == 302:
            url = head_resp.headers.get('Location', url)
            print(f"Redirected to: {url}")
        file_size = int(head_resp.headers.get('content-length', 0))
    except Exception:
        print("cannot get size of dataset, download with single thread...")
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
                    print(f'\nfailed to download a particular chunk, stop downloading: {exc}')
                    sys.exit(1)


def _single_thread_download(url, dest, file_name):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()

            total_size = int(r.headers.get('content-length', 0))

            block_size = 16384  # 16KB

            # if total_size is 0, tqdm switch display model automatically
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


def _download_without_aws(dataset_name: str, dest: str, extensions: list = None):
    if dataset_name == 'widar':
        download_ftp(dataset_name, dest, extensions=extensions)
    elif dataset_name == 'gait':
        download_ftp(dataset_name, dest, extensions=extensions)
    elif dataset_name == 'xrf55':
        os.environ['KAGGLEHUB_CACHE'] = dest
        print(f"os.environ['KAGGLEHUB_CACHE'] is changed to {dest}")
        path = kagglehub.dataset_download("xrfdataset/xrf55-rawdata")
        print("Path to dataset files:", path)
    else:
        raise ValueError(f"no direct download source for '{dataset_name}'")
