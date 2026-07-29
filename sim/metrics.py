import json
from collections import defaultdict


def gini(xs):
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    if s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2 * cum) / (n * s) - (n + 1) / n, 4)


def compute(log_path, burn_in=10, n_agents=None):
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    rows = [r for r in rows if r["tick"] > burn_in]
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event"]].append(r)
    agents = {r["agent_id"] for r in rows if r["agent_id"] != "world"}
    n_agents = n_agents or len(agents) or 1

    pubs = [r for r in by_event["publish"] if r["payload"]["accepted"]]
    oracle = [o for r in pubs for o in r["payload"]["oracle"]]
    author = {r["payload"].get("paper_id"): r["agent_id"]
              for r in by_event["publish"]}
    cites = defaultdict(int)
    for r in by_event["cite"]:
        cites[author.get(r["payload"]["to"], "?")] += 1

    last_active = defaultdict(int)
    for r in rows:
        if r["event"] in ("experiment", "submit", "review_submit"):
            last_active[r["agent_id"]] = max(last_active[r["agent_id"]], r["tick"])
    max_tick = max((r["tick"] for r in rows), default=burn_in)
    exited = {r["agent_id"] for r in by_event["exit"]}
    quiet = [a for a in agents - exited if max_tick - last_active[a] >= 15]

    choose = [r["payload"]["risk"] for r in by_event["choose_problem"]]
    reviews_in = len(by_event["review_accept"]) + len(by_event["review_decline"])

    return {
        "exit_rate": round(len(exited) / n_agents, 4),
        "quiet_quit_rate": round(len(quiet) / n_agents, 4),
        "exploration_index": round(sum(choose) / len(choose), 4) if choose else None,
        "replication_share": round(
            sum(1 for r in pubs if r["payload"].get("replication")) / len(pubs), 4)
            if pubs else None,
        "review_accept_rate": round(
            len(by_event["review_accept"]) / reviews_in, 4) if reviews_in else None,
        "duplication": round(
            sum(1 for r in pubs if r["payload"]["duplicate"]) / len(pubs), 4)
            if pubs else None,
        "citation_gini": gini([cites[a] for a in agents]) if agents else None,
        "false_claim_rate": round(
            sum(1 for o in oracle if o["result"] == "false") / len(oracle), 4)
            if oracle else None,
        "progress_value": sum(r["payload"]["value"] for r in pubs) if pubs else 0,
        "per_agent": {a: {"exited": a in exited, "citations": cites[a]}
                      for a in agents},
    }
