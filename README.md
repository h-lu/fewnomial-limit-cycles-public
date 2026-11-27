# A superquadratic lower bound for limit cycles of planar polynomial vector fields with N monomials and degree at most N − 1

Haibo Lu · Shanghai Institute of Technology · [ORCID](https://orcid.org/0009-0000-2717-5968)

[Main paper](paper/main.pdf) · [Low-term supplement](paper/low-N-supplement.pdf) · [Complete download](paper/fewnomial-r37.zip)

## What the paper proves

The paper asks how many limit cycles can coexist in one real planar polynomial vector field when both the degree and the number of nonzero monomials are limited. Monomials are counted after like terms are collected, separately in the two components. A monomial that occurs in both components counts twice. The total degree is the maximum of the component degrees.

For every integer $N\ge48$, the paper constructs a field with rational coefficients, coprime components, **exactly $N$ monomials** and **degree at most $N-1$**, with at least $\mathcal F(N)>0$ coexisting hyperbolic limit cycles. The explicit integer count satisfies

$$
\mathcal F(N)\ge N^2\log_2 N-50N^2.
$$

The simplified formula gives the leading growth rate; it can be negative at moderate budgets. The actual finite formula already gives 168 cycles at $N=48$. A second result treats the term and degree budgets separately: at every sufficiently large exact degree $\Delta$, the construction gives a lower bound of $2\Delta^2\log_2\Delta-O(\Delta^2)$ using exactly $\lceil 3\Delta/2\rceil$ pure-power terms.

The paper keeps capped-degree and unrestricted-degree comparisons separate. The unrestricted quadratic bound of Gasull and Santana is a different benchmark from the other constructions that fit the same cap. The finite tables say which construction is larger in each comparison; the leading asymptotic estimate does not imply a win at every finite budget.

## Why the construction works

The starting point is a separable Hamiltonian with families of closed ovals. Near a saddle merger, the regular coefficients of the area integral for the outer family add across the two incoming families. This makes it possible to arrange sign changes at several ends with one polynomial perturbation. A block of quadratic folds then increases the local order while retaining disjoint rows of ovals. Jointly varying the vertical potential and the perturbation realizes simple return zeros in a single field. Simple zeros give hyperbolic limit cycles, and the support count shows that the field remains within the prescribed term and degree budgets.

## What to read or download

| File | Contents |
| --- | --- |
| [Main paper](paper/main.pdf) | The main theorems, fold mechanism and comparisons; [LaTeX source](paper/main.tex). |
| [Low-term supplement](paper/low-N-supplement.pdf) | Small-budget examples, return estimates and finite certification details. |
| [Unrestricted fields](paper/unrestricted-fields.pdf) | Separate constructions and certificates without the linear degree cap. |
| [Binary saddle tree](paper/binary-saddle-tree.pdf) | The single-row saddle-tree construction. |
| [Complete final ZIP](paper/fewnomial-r37.zip) | All four PDFs, their source and the reproducibility materials. |
| [Replay ZIP](paper/fewnomial-replay.zip) | The scientific inputs, scripts, reference outputs and source checksums. |
| [Technical guide](paper/CERTIFICATES.md) | Which checks support each result, exact arithmetic details and dependencies. |

The PDFs use named links to their companions. Keep the four PDFs together with their original filenames to preserve those links.

## Reproduce the results

From a checkout, the complete edition is in `paper/`. The root Makefile forwards to that directory, so these commands also work from the repository root:

~~~sh
(cd paper && sha256sum --strict -c SHA256SUMS)
make check
make build
make freeze
~~~

- `make check` replays the finite constructions and comparison calculations. It checks support counts, selected return inequalities, finite allocations and the archived comparison arithmetic. The analytic arguments in the paper prove the all-budget bounds.
- `make build` compiles the four PDFs and checks printed tables, labels and links among them.
- `make freeze` rebuilds the archives and checks the resulting manifest. For a downloaded ZIP, run the same commands from inside its extraction directory without the `cd paper` step.

The checks use Python with the [pinned Python dependencies](paper/requirements.txt), GNU Make, pdfLaTeX, BibTeX and Poppler. The technical guide documents the reference environment and how to install dependencies offline. The independent inputs and reference outputs are in [`inputs/`](paper/inputs/) and [`outputs/`](paper/outputs/); [`evidence/`](paper/evidence/) and [`research/`](paper/research/) contain verification code and witness data used by the replay.

## Stable citation

Haibo Lu, *A superquadratic lower bound for limit cycles of planar polynomial vector fields with N monomials and degree at most N − 1*, preprint, November 27, 2026.

This repository distributes the final manuscript and its reproducibility materials. Use the [`v1.0.1` snapshot](https://github.com/h-lu/fewnomial-limit-cycles-public/tree/v1.0.1) for a stable version, or download the [main PDF](https://github.com/h-lu/fewnomial-limit-cycles-public/blob/v1.0.1/paper/main.pdf) and the [complete reproduction package](https://github.com/h-lu/fewnomial-limit-cycles-public/blob/v1.0.1/paper/fewnomial-r37.zip) from that snapshot.
