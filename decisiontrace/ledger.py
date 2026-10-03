"""Evidence ledger: every number the product shows is a row with a class, a provenance chain,
and the assumptions it depends on.

Evidence classes
----------------
INPUT      what the analyst supplied: experiment configuration, assumptions, population spec
GENERATED  output of simulated agents (aggregated agent responses). Not observation of anyone.
DERIVED    deterministic computation over GENERATED/INPUT evidence (means, leads, stability, thresholds)
OBSERVED   real-world outcome data. None exists in this project; the ledger records that absence explicitly.
"""
from __future__ import annotations

import json
import sqlite3

CLASSES = ("INPUT", "GENERATED", "DERIVED", "OBSERVED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY, title TEXT, question TEXT, run_input_version TEXT,
    current_input_version TEXT, spec_json TEXT);
CREATE TABLE IF NOT EXISTS assumptions (
    experiment_id TEXT, name TEXT, nominal REAL, tested_lo REAL, tested_hi REAL, unit TEXT, description TEXT,
    source TEXT DEFAULT 'analyst input', PRIMARY KEY (experiment_id, name));
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY, experiment_id TEXT, key TEXT, cls TEXT, value_json TEXT, unit TEXT,
    ci_low REAL, ci_high REAL, depends_on_json TEXT, assumptions_json TEXT, note TEXT, details_json TEXT,
    UNIQUE (experiment_id, key));
"""


class Ledger:
    def __init__(self, path=":memory:"):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.executescript(SCHEMA)
        self.eid = None
        self._n = 0

    def start(self, experiment_id, title, question, run_v, cur_v, spec_json):
        self.eid = experiment_id
        self.con.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?)",
                         (experiment_id, title, question, run_v, cur_v, spec_json))

    def add_assumption(self, name, nominal, lo, hi, unit, desc):
        self.con.execute("INSERT OR REPLACE INTO assumptions (experiment_id,name,nominal,tested_lo,tested_hi,unit,description) "
                         "VALUES (?,?,?,?,?,?,?)", (self.eid, name, nominal, lo, hi, unit, desc))

    def add(self, key, cls, value, unit="", ci=(None, None), depends_on=(), assumptions=(), note="", details=None):
        assert cls in CLASSES, cls
        self._n += 1
        evid = f"EV-{self._n:04d}"
        self.con.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (evid, self.eid, key, cls, json.dumps(value), unit, ci[0], ci[1],
                          json.dumps(list(depends_on)), json.dumps(list(assumptions)), note, json.dumps(details or {})))
        return evid

    # ------------------------------------------------------------------ reads
    def _row(self, r):
        if r is None:
            return None
        return {"id": r[0], "key": r[2], "cls": r[3], "value": json.loads(r[4]), "unit": r[5], "ci_low": r[6],
                "ci_high": r[7], "depends_on": json.loads(r[8]), "assumptions": json.loads(r[9]), "note": r[10],
                "details": json.loads(r[11])}

    def get(self, key):
        return self._row(self.con.execute("SELECT * FROM evidence WHERE experiment_id=? AND key=?", (self.eid, key)).fetchone())

    def by_id(self, evid):
        return self._row(self.con.execute("SELECT * FROM evidence WHERE experiment_id=? AND evidence_id=?", (self.eid, evid)).fetchone())

    def find(self, prefix):
        rows = self.con.execute("SELECT * FROM evidence WHERE experiment_id=? AND key LIKE ? ORDER BY evidence_id",
                                (self.eid, prefix + "%")).fetchall()
        return [self._row(r) for r in rows]

    def all(self):
        return [self._row(r) for r in self.con.execute("SELECT * FROM evidence WHERE experiment_id=? ORDER BY evidence_id", (self.eid,))]

    def ids(self):
        return {r[0] for r in self.con.execute("SELECT evidence_id FROM evidence WHERE experiment_id=?", (self.eid,))}

    def assumptions(self):
        rows = self.con.execute("SELECT name,nominal,tested_lo,tested_hi,unit,description,source FROM assumptions WHERE experiment_id=?", (self.eid,)).fetchall()
        return [dict(zip(("name", "nominal", "tested_lo", "tested_hi", "unit", "description", "source"), r)) for r in rows]

    def chain(self, evid):
        """All ancestors of an evidence row, nearest first, with classes. The provenance chain."""
        seen, out, frontier = set(), [], [evid]
        while frontier:
            cur = frontier.pop(0)
            row = self.by_id(cur)
            if row is None or cur in seen:
                continue
            seen.add(cur)
            if cur != evid:
                out.append({"id": row["id"], "key": row["key"], "cls": row["cls"]})
            frontier.extend(row["depends_on"])
        return out

    def classes_in_chain(self, evid):
        return {self.by_id(evid)["cls"]} | {c["cls"] for c in self.chain(evid)}
