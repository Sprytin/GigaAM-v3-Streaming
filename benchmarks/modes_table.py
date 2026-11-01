"""The README's "Choose your mode" tables, built from the benchmark results.

Every number in the tables comes from a results file in ``benchmarks/results``;
nothing is typed by hand. ``--write README.md`` replaces the tables between
the ``<!-- modes:en -->``/``<!-- modes:ru -->`` markers, ``--check README.md``
fails if they differ from what the results give (the test suite runs it).

    python benchmarks/modes_table.py --write README.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
# The run behind each mode: the 898-clip runs and the start-up check.
FILES = {
    "gpu": "v3_e2e_rnnt.json",
    "gpu_memory": "v3_e2e_rnnt_memory.json",
    "cpu": "v3_e2e_rnnt_mode_cpu.json",
    "cpu-preview": "v3_e2e_rnnt_mode_cpu_preview.json",
    "load": "v3_e2e_rnnt_cpu_load.json",
}
COMMANDS = {
    "gpu": '`load_engine("gpu")`',
    "cpu": '`load_engine("cpu")`',
    "cpu-preview": '`load_engine("cpu-preview")`',
}
TEXT = {
    "en": {
        "head": ("", "GPU", "CPU", "CPU + preview"),
        "hardware": ("Needs", "CUDA GPU", "{cores} CPU threads", "{cores} CPU threads"),
        "measured": ("Measured on", "{gpu}", "{cpu}", "{cpu}"),
        "wer": ("WER / CER", "{wer}% / {cer}%"),
        "first": ("First text after a word starts (p50 / p95)", "{p50} / {p95} s"),
        "final": ("Finished phrase after the last word ends (p50 / p95)", "{p50} / {p95} s"),
        "drafts": ("Draft text while a word is spoken", "yes", "no", "yes"),
        "compute": ("Compute per second of audio", "{gpu} s GPU", "{cpu} s CPU"),
        "memory": ("Memory", "{vram} MB GPU", "{rss} MB RAM"),
        "start": ("Start, fresh process, files cached", "{start} s"),
        "command": ("Load it",),
    },
    "ru": {
        "head": ("", "GPU", "CPU", "CPU + черновики"),
        "hardware": ("Нужно", "GPU с CUDA", "{cores} потока CPU", "{cores} потока CPU"),
        "measured": ("Где мерили", "{gpu}", "{cpu}", "{cpu}"),
        "wer": ("WER / CER", "{wer}% / {cer}%"),
        "first": ("Первый текст после начала слова (p50 / p95)", "{p50} / {p95} с"),
        "final": ("Готовая фраза после конца последнего слова (p50 / p95)", "{p50} / {p95} с"),
        "drafts": ("Черновой текст, пока слово звучит", "да", "нет", "да"),
        "compute": ("Счёт на секунду звука", "{gpu} с GPU", "{cpu} с CPU"),
        "memory": ("Память", "{vram} МБ GPU", "{rss} МБ ОЗУ"),
        "start": ("Запуск нового процесса, файлы в кэше", "{start} с"),
        "command": ("Загрузка",),
    },
}


def load(results: Path) -> dict:
    return {key: json.loads((results / name).read_text("utf-8")) for key, name in FILES.items()}


def number(value: float, lang: str, digits: int = 2) -> str:
    """Rounded half up as written in the results (1.595 -> 1.60), ``,`` as the Russian point."""
    text = str(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), ROUND_HALF_UP))
    return text.replace(".", ",") if lang == "ru" else text


def short_cpu(name: str) -> str:
    return re.sub(r"\s+\d+-Core Processor$", "", name).replace("AMD ", "")


def short_gpu(name: str) -> str:
    return name.replace("NVIDIA GeForce ", "")


def table(lang: str, results: Path = RESULTS) -> str:
    runs = load(results)
    text = TEXT[lang]
    modes = ("gpu", "cpu", "cpu-preview")
    rows = [text["head"], ("---",) + ("---:",) * 3]

    def per_mode(key, fill):
        label, *templates = text[key]
        cells = []
        for index, mode in enumerate(modes):
            template = templates[min(index, len(templates) - 1)]
            cells.append(template.format(**fill(mode)))
        rows.append((label, *cells))

    cpu_run = runs["cpu"]
    threads = cpu_run["settings"]["threads"]
    per_mode("hardware", lambda mode: {"cores": threads})
    per_mode(
        "measured",
        lambda mode: {
            "gpu": short_gpu(runs["gpu"]["environment"]["gpu"]),
            "cpu": short_cpu(cpu_run["environment"]["cpu"]),
        },
    )
    per_mode(
        "wer",
        lambda mode: {
            "wer": number(runs[mode]["all"]["wer"]["streaming"], lang),
            "cer": number(runs[mode]["all"]["cer"]["streaming"], lang),
        },
    )
    for key, field in (("first", "first_partial_s"), ("final", "final_lag_s")):
        per_mode(
            key,
            lambda mode, field=field: {
                "p50": number(runs[mode]["all"][field]["p50"], lang),
                "p95": number(runs[mode]["all"][field]["p95"], lang),
            },
        )
    per_mode("drafts", lambda mode: {})
    rows.append(
        (
            text["compute"][0],
            text["compute"][1].format(gpu=number(runs["gpu"]["all"]["rtf"]["streaming"], lang)),
            *(
                text["compute"][2].format(cpu=number(runs[mode]["cpu_rtf"]["streaming_cpu"], lang))
                for mode in modes[1:]
            ),
        )
    )
    rows.append(
        (
            text["memory"][0],
            text["memory"][1].format(
                vram=round(runs["gpu_memory"]["modes"]["streaming"]["peak_mb"])
            ),
            *(text["memory"][2].format(rss=round(runs[mode]["peak_rss_mb"])) for mode in modes[1:]),
        )
    )
    start = text["start"][1].format(
        start=number(runs["load"]["median"]["cached"]["ready_s"], lang, 1)
    )
    rows.append((text["start"][0], "—", start, start))  # not measured on the GPU
    rows.append((text["command"][0], *(COMMANDS[mode] for mode in modes)))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def replace(readme: str, lang: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- modes:{lang} -->\n)(?:.*?\n)?(<!-- /modes:{lang} -->)", re.S)
    if not pattern.search(readme):
        raise SystemExit(f"README has no <!-- modes:{lang} --> block")
    return pattern.sub(lambda match: match.group(1) + body + "\n" + match.group(2), readme)


def render(readme: str, results: Path = RESULTS) -> str:
    for lang in TEXT:
        readme = replace(readme, lang, table(lang, results))
    return readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", type=Path, help="update the tables in this README")
    action.add_argument("--check", type=Path, help="fail if this README's tables are stale")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.write:
        args.write.write_text(render(args.write.read_text("utf-8"), args.results), "utf-8")
    elif args.check:
        current = args.check.read_text("utf-8")
        if render(current, args.results) != current:
            sys.exit(f"{args.check}: the mode tables differ from {args.results}; run --write")
    else:
        for lang in TEXT:
            print(table(lang, args.results), end="\n\n")


if __name__ == "__main__":
    main()
