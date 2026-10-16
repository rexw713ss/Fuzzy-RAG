"""Build a local indexed passage store from the original, fingerprinted corpus."""

import argparse
from pathlib import Path
import sqlite3

from main_logic.corpus import iter_passages
from experiment_ver2.followup.io import digest


def build(source, output, expected_sha256):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("refuse to overwrite passage store")
    if digest(source) != expected_sha256:
        raise ValueError("corpus checksum does not match the declared source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as db:
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.execute("CREATE TABLE passages(id TEXT PRIMARY KEY,title TEXT NOT NULL,text TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES ('status','building')")
        db.execute("INSERT INTO metadata VALUES ('source_sha256',?)", (expected_sha256,))
        count = 0
        for doc_id, title, text in iter_passages(source):
            db.execute("INSERT INTO passages VALUES (?,?,?)", (doc_id, title, text))
            count += 1
            if count % 10000 == 0:
                db.commit()
        db.execute("INSERT INTO metadata VALUES ('n_passages',?)", (str(count),))
        db.execute("UPDATE metadata SET value='complete' WHERE key='status'")
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(f"Indexed {build(a.corpus, a.out, a.sha256)} passages")


if __name__ == "__main__":
    main()
