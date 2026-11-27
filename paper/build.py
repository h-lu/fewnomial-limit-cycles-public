"""Build the four PDFs in an isolated relative directory. No live network dependencies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from bundle_replay import DOCUMENTS, build_replay

root = Path(__file__).resolve().parent
build = root/"work"/"build"
build.mkdir(parents=True,exist_ok=True)
build_replay(root)
env=os.environ.copy()
env.update(SOURCE_DATE_EPOCH="1795478400",TZ="UTC",TEXINPUTS="./:"+str(root)+"/:",BIBINPUTS=str(root)+"/:")
def invoke(args):
    done=subprocess.run(args,cwd=build,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if done.returncode:
        print(done.stdout[-5000:],file=sys.stderr)
        raise SystemExit(done.returncode)
def latex(stem):
    invoke(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", stem + ".tex"])


for stem in DOCUMENTS:
    latex(stem)
    invoke(["bibtex", stem])
    for _ in range(3):
        latex(stem)

for stem in DOCUMENTS:
    log=(build/(stem+".log")).read_text(errors="replace")
    if "There were undefined references" in log or "There were undefined citations" in log:
        print("\n".join(l for l in log.splitlines() if "undefined" in l),file=sys.stderr)
        raise SystemExit("Unresolved cross-references")
    shutil.copyfile(build/(stem+".pdf"),root/(stem+".pdf"))
    print(subprocess.check_output(["pdfinfo",str(root/(stem+".pdf"))],text=True).split("Pages:")[1].splitlines()[0].strip(),stem+".pdf")
