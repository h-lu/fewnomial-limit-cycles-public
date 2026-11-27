"""Optional standalone typesetting tests on fresh extracts of the scientific archive."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pypdf import PdfReader


ARCHIVE = Path(__file__).resolve().parents[1] / "fewnomial-replay.zip"
sys.path.insert(0, str(ARCHIVE.parent))
from bundle_replay import DOCUMENTS as STEMS


def compile_article(folder, engine, stem, env):
    def run(*args):
        result = subprocess.run(args, cwd=folder, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        if result.returncode:
            raise RuntimeError(" ".join(args) + "\n" + result.stdout[-4000:])
    latex = [engine, "-no-shell-escape", "-halt-on-error", "-interaction=batchmode", stem + ".tex"]
    run(*latex)
    run("bibtex", stem)
    run(*latex)
    run(*latex)
    log = (folder / (stem + ".log")).read_text(errors="replace")
    assert "undefined references" not in log and "undefined citations" not in log, stem
    return log


def check_variant(engine, with_archive):
    with TemporaryDirectory(prefix="fewnomial-tex-") as tmp:
        inputs_per_article = []
        for stem in STEMS:
            folder = Path(tmp) / stem
            folder.mkdir()
            # Each directory has only this article's literal inputs and bibliography.
            with ZipFile(ARCHIVE) as archive:
                pending = [stem + ".tex"]
                names = {"references.bib"}
                while pending:
                    name = pending.pop()
                    if name in names:
                        continue
                    names.add(name)
                    for key in re.findall(r"\\(?:inputrows|input)\{([^{}#]+)\}",
                                          archive.read(name).decode()):
                        pending.append(key if key.endswith(".tex") else key + ".tex")
                inputs_per_article.append(len(names))
                for name in sorted(names):
                    archive.extract(name, folder)
            if with_archive:
                shutil.copyfile(ARCHIVE, folder / ARCHIVE.name)
            env = dict(os.environ, TEXINPUTS=".:", BIBINPUTS=".:", TEXMFOUTPUT=str(folder),
                       SOURCE_DATE_EPOCH="1795478400", TZ="UTC")
            assert not any(folder.glob("*.pdf")) and not any(folder.glob("*.aux"))
            compile_article(folder, engine, stem, env)
            if engine == "latex":
                assert (folder / (stem + ".dvi")).read_bytes()[:2] == bytes([247, 2])
                continue
            pdf = PdfReader(folder / (stem + ".pdf"))
            assert not pdf.attachments
            for page in pdf.pages:
                for ref in page.get("/Annots", []):
                    action = ref.get_object().get("/A")
                    if action and action["/S"] == "/GoToR":
                        assert Path(action["/F"]).name == action["/F"]
                        assert action["/F"] in {s + ".pdf" for s in STEMS if s != stem}
                        assert isinstance(action["/D"], str) and action["/D"]
                    else:
                        assert not action or action["/S"] in ("/GoTo", "/URI")
            if engine == "pdflatex":
                canonical = ARCHIVE.parent / (stem + ".pdf")
                if canonical.is_file():
                    assert (folder / canonical.name).read_bytes() == canonical.read_bytes()
        fmt = "DVI" if engine == "latex" else "PDF"
        print(f"{engine} with archive={with_archive}: all {len(STEMS)} {fmt} documents built separately with resolved references.")
        print("  Required source files: " + ", ".join(f"{s} {n}" for s, n in zip(STEMS, inputs_per_article)))
        if engine == "pdflatex" and all((ARCHIVE.parent / (s + ".pdf")).is_file() for s in STEMS):
            print("  Each isolated pdfLaTeX output matches its canonical PDF byte for byte.")
        if fmt == "PDF":
            print("  No attachments. Companion links use named destinations in sibling PDFs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engines", nargs="*", help="engines to test (default: pdflatex)")
    args = parser.parse_args()
    for engine in args.engines or ["pdflatex"]:
        if engine not in ("pdflatex", "lualatex", "latex"):
            parser.error(f"unsupported test engine {engine}")
        if not shutil.which(engine):
            parser.error(f"no installed {engine} executable")
        for present in (False, True):
            check_variant(engine, present)
