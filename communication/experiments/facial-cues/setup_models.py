"""Download only pinned, checksummed weights; never runs during inference."""
import hashlib
import json
from pathlib import Path
import httpx

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'data' / 'face-models'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for item in json.loads((HERE / 'models.json').read_text())['files']:
            path = (DEST if item['model'] else HERE) / item['name']
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
                print(item['name'], 'verified'); continue
            content = bytearray()
            with client.stream('GET', item['url']) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > item['bytes']: raise ValueError('Download larger than pinned asset')
            if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Checksum mismatch: ' + item['name'])
            path.with_suffix(path.suffix + '.tmp').write_bytes(content)
            path.with_suffix(path.suffix + '.tmp').replace(path)
            print(item['name'], 'installed and verified')

if __name__ == '__main__': main()
