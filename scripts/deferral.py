"""Appendix G2: deferral language vs follow-through, archived lived worlds.

    .venv/bin/python scripts/deferral.py

No API calls. For each accepted paper in the 16 completed drift worlds:
does its title+body defer confirmation (pre-registered regexes), and did
its author ever run a later intervention touching any variable of any of
its claims?

The per-run llm.jsonl aggregates 8 worlds, and custom_ids (agent-tick)
repeat across them, so a write action is matched to its paper by custom_id
+ board-arm clause in the circular + canonical claim keys against the
publish event's oracle keys. Ambiguous matches are counted and excluded.
"""
import json
import re
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.verify import canonical_key, normalize_claim         # noqa: E402
from sim.verify import extract_node                           # noqa: E402

DIRS = ["runs/20260801T161157Z-drift", "runs/20260801T161215Z-drift"]
CLAUSE = {"PROXY": "weighs the number of items",
          "VERIFIED": "commissions independent replication"}

DEFER = [re.compile(r, re.I) for r in [
    r"further (investigation|stud(y|ies)|work|research|observation|experiments?|confirmation)",
    r"future (work|stud(y|ies)|research|investigation)",
    r"(requires?|warrants?|needs?|merits?) (further|additional|more|deeper)",
    r"follow-?up",
]]
DEFER2 = [re.compile(r, re.I) for r in [
    r"(to|should|must|could|would|will) (be )?(confirm|verif|validat)\w*",
    r"(confirmation|verification|validation) (is|are|would|will|remains?) (needed|required|pending|warranted)",
]]


def keys_of(resp):
    out = set()
    for c in resp.get("claims") or []:
        try:
            out.add(canonical_key(normalize_claim(c)))
        except Exception:
            pass
    return out


def main():
    res = {}
    for arm in ("PROXY", "VERIFIED"):
        # accepted papers with claim keys, per world
        accepted = []          # dicts: world, pid, agent, tick, keys
        interventions = []     # (world, agent, tick, vars)
        for base in DIRS:
            for d in sorted(glob(f"{base}/pair*-{arm}")):
                submit_meta = {}
                for line in open(d + "/log.jsonl"):
                    r = json.loads(line)
                    ev, p = r["event"], r["payload"]
                    if ev == "submit" and p["admissible"]:
                        submit_meta[p["paper_id"]] = (r["agent_id"], r["tick"])
                    elif ev == "publish" and p["accepted"]:
                        agent, tick = submit_meta.get(p["paper_id"], (None, None))
                        if agent is None:
                            continue
                        accepted.append({
                            "world": d, "base": base, "pid": p["paper_id"],
                            "agent": agent, "tick": tick,
                            "keys": {o["key"] for o in p["oracle"]}})
                    elif ev == "experiment" and p["kind"] == "intervene":
                        interventions.append(
                            (d, r["agent_id"], r["tick"],
                             set(p["measured"]) | set(p["targets"])))
        # write actions by (base, cid), arm-filtered
        writes = {}
        for base in DIRS:
            for line in open(base + "/llm.jsonl"):
                r = json.loads(line)
                if CLAUSE[arm] not in (r.get("user") or ""):
                    continue
                resp = r.get("response")
                if isinstance(resp, str):
                    try:
                        resp = json.loads(resp)
                    except Exception:
                        continue
                if not (isinstance(resp, dict)
                        and resp.get("action") == "write"):
                    continue
                writes.setdefault((base, r["custom_id"]), []).append(resp)
        n_def = n_def2 = n_follow = unmatched = ambiguous = 0
        matched = 0
        for pap in accepted:
            cands = [w for w in
                     writes.get((pap["base"], f"{pap['agent']}-t{pap['tick']}"),
                                [])
                     if keys_of(w) == pap["keys"]]
            texts = {json.dumps([w.get("title"), w.get("body")])
                     for w in cands}
            if not cands:
                unmatched += 1
                continue
            if len(texts) > 1:
                ambiguous += 1
                continue
            matched += 1
            w = cands[0]
            blob = f"{w.get('title') or ''} {w.get('body') or ''}"
            hit = any(rx.search(blob) for rx in DEFER)
            n_def2 += any(rx.search(blob) for rx in DEFER2)
            if not hit:
                continue
            n_def += 1
            cvars = {extract_node(c.get(k))
                     for c in (w.get("claims") or [])
                     for k in ("cause", "effect") if c.get(k)}
            cvars |= {extract_node(x) for c in (w.get("claims") or [])
                      for x in (c.get("causes") or [])}
            cvars.discard(None)
            if any(a == pap["agent"] and t > pap["tick"] and cvars & vs
                   for (dd, a, t, vs) in interventions
                   if dd == pap["world"]):
                n_follow += 1
        res[arm] = {
            "accepted": len(accepted), "matched": matched,
            "unmatched": unmatched, "ambiguous": ambiguous,
            "deferral_flagged": n_def,
            "deferral_share": round(n_def / max(matched, 1), 3),
            "modal_confirm_flagged": n_def2,
            "modal_share": round(n_def2 / max(matched, 1), 3),
            "followed_through": n_follow,
            "follow_through_rate": round(n_follow / max(n_def, 1), 3),
            "world_interventions_total": len(interventions)}
    print(json.dumps(res, indent=2))
    Path("runs/deferral-g2.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
