import os
import ftplib

from tqdm import tqdm
from urllib.parse import urlparse, unquote
from .load_preset import load_api


def download_ftp(dataset_name: str, dest: str):
    url = load_api(dataset_name)
    parsed = urlparse(url)
    ftp_server = parsed.hostname
    ftp_port = parsed.port
    ftp_user = parsed.username
    ftp_pass = parsed.password
    ftp_root_path = unquote(parsed.path)

    try:
        ftp = ftplib.FTP()
        ftp.connect(ftp_server, ftp_port)
        ftp.login(ftp_user, ftp_pass)

        ftp.encoding = 'utf-8'
        ftp.set_pasv(True)
        print(f"prepare to download from: {ftp_root_path}")

        try:
            ftp.cwd(ftp_root_path)
        except ftplib.error_perm as e:
            print(f"Error: cannot dive into: '{ftp_root_path}'.")
            raise e

        _download_current_dir(ftp, dest)

        ftp.quit()
        print("\nAll files downloaded")

    except Exception as e:
        raise e


def _download_current_dir(ftp, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    try:
        filenames = ftp.nlst()
    except ftplib.error_perm:
        return

    for filename in filenames:
        if filename in ('.', '..'):
            continue

        local_path = os.path.join(local_dir, filename)

        try:
            ftp.cwd(filename)
            _download_current_dir(ftp, local_path)
            ftp.cwd('..')

        except ftplib.error_perm:
            try:
                try:
                    file_size = ftp.size(filename)
                except:
                    file_size = None

                with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, leave=False) as pbar:
                    with open(local_path, 'wb') as f:
                        def download_callback(data):
                            f.write(data)
                            pbar.update(len(data))

                        ftp.retrbinary('RETR ' + filename, download_callback)

            except Exception as e:
                raise e
