"""Strict JSON, immutable output directories and content fingerprints."""

import hashlib
import json
from pathlib import Path


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse(text):
    return json.loads(text, object_pairs_hook=_object,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {v}")))


def read_json(path):
    return parse(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [parse(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
                          encoding="utf-8", newline="\n")


def new_directory(path, inputs=()):
    path = Path(path).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if path == source or path in source.parents or (source.is_dir() and source in path.parents):
            raise ValueError("output directory must be separate from all inputs")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("refuse to overwrite a nonempty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def seal(directory):
    directory = Path(directory)
    write_json(directory / "SHA256.json", {p.relative_to(directory).as_posix(): digest(p)
               for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "SHA256.json"})
