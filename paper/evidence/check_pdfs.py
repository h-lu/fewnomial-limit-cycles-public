"""Check label pages, printed companion references, PDF actions and the scientific ZIP."""
from collections import Counter
from csv import DictReader
from hashlib import sha256
from pathlib import Path
import re
from urllib.parse import urlsplit
from zipfile import ZipFile
from io import BytesIO
import json
import subprocess
import sys

from pypdf import PdfReader
from pypdf.generic import ArrayObject, NumberObject


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bundle_replay import DOCUMENTS as STEMS


def groups(line):
    result = []
    depth = 0
    for i, char in enumerate(line):
        if char in "{}" and (i == 0 or line[i-1] != "\\"):
            if char == "{":
                if depth == 0:
                    start = i + 1
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    result.append(line[start:i])
    assert depth == 0, line
    return result


def verify_target(reader, dest):
    if isinstance(dest, str):
        key = dest[1:] if dest.startswith("/") else dest
        assert key in reader.named_destinations, key
        page = reader.get_destination_page_number(reader.named_destinations[key])
    else:
        assert isinstance(dest, ArrayObject) and len(dest) >= 2, dest
        first = dest[0]
        if isinstance(first, (int, NumberObject)):
            page = int(first)
        else:
            page = reader.get_page_number(first.get_object())
    assert page is not None and 0 <= page < len(reader.pages)
    return page


def article_sources(stem):
    """Follow only literal manuscript inputs, including generated table rows."""
    seen = set()
    pending = [stem + ".tex"]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        source = (ROOT / name).read_text()
        for item in re.findall(r"\\(?:inputrows|input)\{([^{}#]+)\}", source):
            pending.append(item if item.endswith(".tex") else item + ".tex")
    return seen


def check_companion_numbers(stem, labels):
    other_labels = {}
    for other, label_map in labels.items():
        if other != stem:
            assert not (other_labels.keys() & label_map.keys()), "Ambiguous companion label"
            other_labels.update(label_map)
    refs = []
    targets = set()
    for name in sorted(article_sources(stem)):
        source = (ROOT / name).read_text()
        refs.extend(re.findall(r"\\othernumber\{([^{}]+)\}\{([^{}]+)\}",
                               source))
        for other, dest, key, printed in re.findall(
                r"\\paperlink\{([^{}]+)\}\{([^{}]+)\}\{([^{}]+)\}\{([^{}]+)\}", source):
            assert other != stem and labels[other][key][3] == dest, (stem, key, dest)
            refs.append((key, printed))
            targets.add((other + ".pdf", dest))
    assert refs, stem
    for key, printed_number in refs:
        assert other_labels[key][0] == printed_number, (stem, key, printed_number)
    if stem == "main":
        required = {"sup:sources", "sup:renewal-support", "sup:mechanical-budget", "sup:focus",
                    "sup:quartic", "sec:seed-replication", "eq:capped-focus-return", "tab:capped-exterior-signs", "prop:quintic",
                    "thm:stationary-turn", "prop:783-witness", "sup:reproduction"}
        assert required <= {k for k, _ in refs}
    print(f"{stem}.pdf: {len(refs)} printed companion references agree with "
          f"{len(set(refs))} distinct LaTeX labels in the companion documents.")
    return targets


def page_text(file, page):
    return subprocess.check_output(["pdftotext", "-layout", "-f", str(page),
                                    "-l", str(page), str(ROOT / file), "-"], text=True)


def check_table_page(readers, labels):
    main = readers["main.pdf"]
    page = 1 + verify_target(main, labels["main"]["tab:small"][3])
    body = page_text("main.pdf", page)
    classical = [(1,3),(2,3),(4,5),(6,5),(9,7),(25,8),(35,8)]
    pub_capped = [r"1 \(3\)", r"3 \(3\)", r"4a \(3\)", r"5a \(3\)", r"5 \(6\)", r"24 \(4\)", r"24p \(4\)"]
    for n,(count,degree) in enumerate(classical, start=4):
        elementary = (n-2)**2 // 4
        extra = {5:(4,4), 6:(6,3), 7:(10,5), 8:(16,7)}.get(n, ("—", "—"))
        row = [str(n), str(elementary), str(count), str(degree), *map(str, extra), pub_capped[n-4]]
        cells = r"[ \t]+".join(row)
        assert re.search(rf"^[ \t]*{cells}[ \t]*$", body, re.MULTILINE), (n,page,body)
    flat = " ".join(body.split())
    for word in ("Main text", "Published", "Supplement",
                 "S2.1", "S2.2", "exterior sign enclosures", "Table " + labels["low-N-supplement"]["tab:capped-exterior-signs"][0]):
        assert word in flat, (page, word)
    supplement = readers["low-N-supplement.pdf"]
    for key,text in (("eq:capped-focus-return","positive roots are simple"),
                     ("tab:capped-exterior-signs","exterior test energies")):
        p = 1 + verify_target(supplement, labels["low-N-supplement"][key][3])
        assert text in " ".join(page_text("low-N-supplement.pdf",p).split()),key
    print(f"Low-support table on p. {page}: seven article rows, seven capped published comparisons, four additional capped fields in the table body, "
          "degrees, equation and exterior sign table verified.")



def check_centers(readers, labels):
    file = "unrestricted-fields.pdf"
    label = labels["unrestricted-fields"]["tab:focus-centers"]
    page = 1 + verify_target(readers[file], label[3])
    rows = re.findall(r"^\s*([JT]\d{2})[ \t]+([0-9a-f]{73})[ \t]*\n[ \t]+([0-9a-f]+)[ \t]*$",
                      page_text(file, page), re.MULTILINE)
    keys = [f"J{i:02d}" for i in range(1, 11)] + [f"T{i:02d}" for i in range(1, 8)]
    data = (ROOT/"research/tail_interpolation/brackets.txt").read_text().splitlines()
    assert data[0] == "exp" and data[11] == "linear" and len(data) == 19
    assert [key for key, _, _ in rows] == keys
    assert [a+b for _, a, b in rows] == data[1:11]+data[12:]
    for section, upper in ((rows[:10], 2**576-2**464), (rows[10:], 5*2**574)):
        centers = [int(a+b, 16) for _, a, b in section]
        radius = 2**64
        assert centers[0]-radius > 2**486 and centers[-1]+radius < upper
        assert all(a+radius < b-radius for a, b in zip(centers, centers[1:]))
    print(f"Table {label[0]} on p. {page}: all 17 printed hexadecimal numerators match the ordered "
          "bracket input digit for digit; both families satisfy the exact domain and separation bounds.")


def check_budget_table(reader, labels):
    keys = ("tab:mechanism-breakdown", "tab:fold", "tab:small")
    pages = []
    for i, key in enumerate(keys, start=1):
        number, printed, _, dest, *_ = labels[key]
        page = 1 + verify_target(reader, dest)
        assert (number, printed) == (str(i), str(page)), key
        pages.append(page)
    assert pages == sorted(set(pages))
    body = page_text("main.pdf", pages[0])
    strip = labels["prop:strip-product"][0]
    fixed = labels["prop:fixed-potential"][0]
    main = labels["cor:linear-cap"][0]
    joint = labels["thm:joint-main"][0]
    expected = [(f"Separate strips (Proposition {strip})", "1/4", "1/4"),
                (f"Shared wells, fixed potential (Proposition {fixed})", "1", "1"),
                (f"Shared wells, joint variation (Theorems {main} and {joint})", "1", "2")]
    for row in expected:
        assert re.search(r"^\s*" + r"[ \t]+".join(map(re.escape, row)) + r"[ \t]*$",
                         body, re.MULTILINE), row
    compact = "".join(body.split()).replace("∆", "Δ")
    for text in ("N2log2N", "Δ2log2Δ", "construction", "saturation"):
        assert text.lower() in compact.lower(), text
    records = {v["N"]: v for v in json.loads((ROOT/"outputs/finite-stages.json").read_text())}
    budgets = (48,256,512,1024,4096,8192,32768,65536,524288,1048576)
    body = page_text("main.pdf", pages[1])
    for n in budgets:
        row = records[n]
        fold = row["fold"]
        values = (n, fold["r"], fold["D"], fold["cycles"], row["stationary"]["cycles"],
                  row["separate_strips"]["cycles"])
        pattern = r"^[ \t]*" + r"[ \t]+".join(f"{v:,}" for v in values) + r"[ \t]*$"
        assert re.search(pattern, body, re.MULTILINE), (n, pages[1])
    print(f"Table 1 on p. {pages[0]}: three internal constructions, two normalized coefficients each.")
    print(f"Table 2 on p. {pages[1]}: ten exact-budget rows, folded degrees and three counts per row.")


def check_witness_and_grid():
    def document(stem):
        body = subprocess.check_output(["pdftotext", "-layout", str(ROOT / (stem + ".pdf")), "-"],
                                       text=True)
        return " ".join(body.split())

    main, supplement = map(document, ("main", "low-N-supplement"))
    for row in ("4 (1, 3) (2, 4) (4, 7)", "5 (2, 3) (4, 4) (8, 7)"):
        assert row in supplement, row
    with (ROOT / "outputs/783-brackets.tsv").open() as data:
        signs = Counter(int(row["sign"]) for row in DictReader(
            (line for line in data if not line.startswith("#")), delimiter="\t"))
    strips = 29
    cycles = strips * (signs[1] + signs[-1])
    assert (set(signs), cycles) == ({1, -1}, 783)
    assert f"{strips}({signs[1]} + {signs[-1]}) = {cycles} hyperbolic cycles with 32 terms at degree 30" in main
    assert f"{cycles} hyperbolic cycles at degree 31" in supplement
    with (ROOT / "outputs/dyadic-comparisons.tsv").open() as data:
        rows = list(DictReader(data, delimiter="\t"))
    assert [int(row["j"]) for row in rows] == list(range(6, 21))
    first = next(row for row in rows if int(row["folded"]) > int(row["classical_preprint"]))
    assert int(first["N"]) == 2**16
    for key in ("folded", "classical_preprint"):
        assert f"{int(first[key]):,}" in supplement
    assert f"{int(first['classical_preprint']):,}" not in main
    print("Printed lifts, stationary example at degrees 30 and 31, and conditional dyadic comparison verified.")


def main():
    replay = (ROOT / "fewnomial-replay.zip").read_bytes()
    with ZipFile(BytesIO(replay)) as z:
        assert z.testzip() is None
        manifest = z.read("SHA256SUMS").decode().splitlines()
        expected = {"SHA256SUMS"}
        for line in manifest:
            digest, name = line.split("  ", 1)
            expected.add(name)
            assert sha256(z.read(name)).hexdigest() == digest
            assert z.read(name) == (ROOT / name).read_bytes()
        assert set(z.namelist()) == expected and len(z.namelist()) == len(expected)
    print(f"Independent scientific ZIP: {len(expected)} members, SHA-256 {sha256(replay).hexdigest()}")
    readers = {stem+".pdf": PdfReader(ROOT / (stem+".pdf")) for stem in STEMS}
    labels = {}
    for stem in STEMS:
        labels[stem] = {}
        for line in (ROOT / "work/build" / (stem + ".aux")).read_text().splitlines():
            if line.startswith(r"\newlabel{"):
                label, details = groups(line[len(r"\newlabel"):])
                labels[stem][label] = groups(details)
    for file, reader in readers.items():
        assert not reader.attachments, file
        counts = Counter()
        targets = set()
        for page in reader.pages:
            for ref in page.get("/Annots", []):
                annotation = ref.get_object()
                if annotation.get("/Subtype") != "/Link":
                    continue
                if "/Dest" in annotation:
                    verify_target(reader, annotation["/Dest"])
                    counts["internal"] += 1
                    continue
                action = annotation["/A"]
                kind = action["/S"]
                if kind == "/GoTo":
                    verify_target(reader, action["/D"])
                    counts["internal"] += 1
                elif kind == "/GoToR":
                    other, dest = action["/F"], action["/D"]
                    assert other in readers and other != file and isinstance(dest, str), (file, action)
                    verify_target(readers[other], dest)
                    targets.add((other, dest))
                    counts["companion links"] += 1
                elif kind == "/URI":
                    parsed = urlsplit(action["/URI"])
                    assert parsed.scheme in ("https", "http") and parsed.netloc
                    counts["web URI (syntax)"] += 1
                else:
                    raise ValueError(f"Unexpected PDF link action {kind}")
        stem = Path(file).stem
        expected = check_companion_numbers(stem, labels)
        assert targets == expected, (file, targets ^ expected)
        for label, (number, printed, title, destination, *_) in labels[stem].items():
            assert verify_target(reader, destination) == int(printed)-1, (file, label)
            counts["printed labels"] += 1
        log = (ROOT / "work/build" / (stem+".log")).read_text()
        assert not re.search(r"undefined references|undefined citations|multiply defined|Rerun to get cross-references right|Missing character|Overfull \\[hv]box", log, re.I)
        print(f"{file}: {len(reader.pages)} pages, {len(reader.named_destinations)} named destinations; "
              + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "; no embedded files.")
    check_table_page(readers, labels)
    check_budget_table(readers["main.pdf"], labels["main"])
    check_witness_and_grid()
    command = "python3 finite.py --stages 512"
    check_centers(readers, labels)
    assert command in (ROOT/"CERTIFICATES.md").read_text()
    current = json.loads(subprocess.check_output(command.split(), cwd=ROOT, text=True))
    stored = json.loads((ROOT / "outputs/finite-stages.json").read_text())
    assert current == next(row for row in stored if row["N"] == 512)
    print("The example command in CERTIFICATES.md reproduces the complete stored N=512 allocation.")
    print("All label pages and link actions checked. External URI validation covers syntax.")


if __name__ == "__main__":
    main()
