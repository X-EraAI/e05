"""Install pinned OIDN with Blackwell support into E05's isolated SAPIEN package."""
import hashlib
import sys
import sysconfig
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2.3.3'
NAME = f'oidn-{VERSION}.x86_64.linux'
URL = f'https://github.com/RenderKit/oidn/releases/download/v{VERSION}/{NAME}.tar.gz'
SHA256 = '3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f'


def main():
    prefix = Path(sys.prefix).resolve()
    if not prefix.is_relative_to(ROOT):
        raise RuntimeError('Run with the isolated E05 environment; refusing to modify an external prefix')
    package = Path(sysconfig.get_paths()['purelib']) / 'sapien'
    loader = package / '_oidn_tricks.py'
    source = loader.read_text()
    if '2.0.1' not in source and VERSION not in source:
        raise RuntimeError('Unrecognized SAPIEN OIDN loader; review compatibility first')
    archive = ROOT / 'downloads' / f'{NAME}.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=90) as response, partial.open('wb') as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError(f'OIDN archive checksum mismatch: {archive}')
    destination = package / 'oidn_library'
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        for name in ('libOpenImageDenoise', 'libOpenImageDenoise_core', 'libOpenImageDenoise_device_cuda'):
            filename = f'{name}.so.{VERSION}'
            member = bundle.getmember(f'{NAME}/lib/{filename}')
            if not member.isfile():
                raise RuntimeError(f'Expected regular library: {member.name}')
            (destination / filename).write_bytes(bundle.extractfile(member).read())
        notices = destination / f'licenses-{VERSION}'
        notices.mkdir(exist_ok=True)
        for member in bundle.getmembers():
            if member.isfile() and member.name.startswith(f'{NAME}/doc/') and member.name.endswith('.txt'):
                (notices / Path(member.name).name).write_bytes(bundle.extractfile(member).read())
    loader.write_text(source.replace('2.0.1', VERSION))
    print(f'Installed OIDN {VERSION} from pinned upstream archive into {destination}')


if __name__ == '__main__':
    main()
