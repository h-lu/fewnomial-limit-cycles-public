"""Build the self-contained, deterministic scientific companion archive."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


DOCUMENTS = ("main", "low-N-supplement", "unrestricted-fields", "binary-saddle-tree")


def write_archive(archive, entries):
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as out:
        for name, data in sorted(entries.items()):
            info = ZipInfo(name, (2026, 11, 23, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            out.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)


def build_replay(root):
    top = ["Makefile", "requirements.txt", "references.bib", "CERTIFICATES.md",
           *(stem + ".tex" for stem in DOCUMENTS)]
    paths = {root / name for name in top} | set(root.glob("*.py"))
    for directory in ("inputs", "research", "tex", "outputs", "primary"):
        paths.update(p for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    paths.update((root / "evidence").glob("check_*.py"))
    entries = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(paths)}
    entries["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in entries.items()
    ).encode("ascii")
    archive = root / "fewnomial-replay.zip"
    write_archive(archive, entries)
    return archive


def write_release_manifest(root):
    """Add the compiled articles and the source ZIP to the archive's source inventory."""
    archive = root / "fewnomial-replay.zip"
    with ZipFile(archive) as source:
        rows = source.read("SHA256SUMS").decode("ascii").splitlines()
    for path in [*(root / (stem + ".pdf") for stem in DOCUMENTS), archive]:
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    rows.sort(key=lambda row: row.split("  ", 1)[1])
    manifest = ("\n".join(rows) + "\n").encode("ascii")
    (root / "SHA256SUMS").write_bytes(manifest)
    files = {row.split("  ", 1)[1]: (root / row.split("  ", 1)[1]).read_bytes() for row in rows}
    write_archive(root / "fewnomial-r37.zip", {**files, "SHA256SUMS": manifest})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", action="store_true",
                        help="freeze the verified source archive and all four PDFs")
    root = Path(__file__).resolve().parent
    if parser.parse_args().release_manifest:
        write_release_manifest(root)
    else:
        build_replay(root)
