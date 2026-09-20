from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import networkx as nx
import numpy as np
import opendssdirect as dss
from scipy.optimize import linear_sum_assignment

from restoration.environment import FeederSpec, Switch, deterministic_split


SOURCE_COMMIT = "5005c668a72d20775f4c2d060feebb2866ba1d38"


def bus_name(value: str) -> str:
    return value.split(".")[0].strip().lower()


def compile_case(master: Path) -> None:
    dss.Basic.ClearAll()
    dss.Text.Command(f"compile [{master.resolve()}]")
    dss.Text.Command("set controlmode=off")
    dss.Text.Command("solve")


def extract_graph(master: Path) -> tuple[nx.Graph, dict[str, float]]:
    compile_case(master)
    graph = nx.Graph()
    graph.add_nodes_from(bus_name(x) for x in dss.Circuit.AllBusNames())
    for element in dss.Circuit.AllElementNames():
        lower = element.lower()
        if not (lower.startswith("line.") or lower.startswith("transformer.")):
            continue
        dss.Circuit.SetActiveElement(element)
        buses = [bus_name(x) for x in dss.CktElement.BusNames()]
        if len(buses) >= 2 and buses[0] and buses[1] and buses[0] != buses[1]:
            graph.add_edge(buses[0], buses[1], element=element)
    load_kw: dict[str, float] = {}
    for load in dss.Loads.AllNames():
        dss.Loads.Name(load)
        buses = dss.CktElement.BusNames()
        if buses:
            b = bus_name(buses[0])
            load_kw[b] = load_kw.get(b, 0.0) + float(dss.Loads.kW())
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    if not nx.is_connected(graph):
        largest = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()
        load_kw = {k: v for k, v in load_kw.items() if k in graph}
    return graph, load_kw


def parse_123_switches(master: Path, graph: nx.Graph) -> list[tuple[str, str, str]]:
    text = master.read_text(encoding="utf-8", errors="ignore")
    line_pairs: dict[str, tuple[str, str]] = {}
    for raw in text.splitlines():
        match = re.search(r"(?i)^\s*new\s+line\.([\w-]+).*?bus1=([^\s!]+).*?bus2=([^\s!]+)", raw)
        if match:
            line_pairs[match.group(1).lower()] = (bus_name(match.group(2)), bus_name(match.group(3)))
    controls: list[tuple[str, str, str]] = []
    for raw in text.splitlines():
        match = re.search(r"(?i)^\s*new\s+swtcontrol\.([\w-]+).*?switchedobj=line\.([\w-]+)", raw)
        if not match:
            continue
        control, line = match.group(1), match.group(2).lower()
        if line in line_pairs:
            u, v = line_pairs[line]
        else:
            compile_case(master)
            if dss.Lines.Name(line) == 0:
                continue
            buses = dss.CktElement.BusNames()
            u, v = bus_name(buses[0]), bus_name(buses[1])
        if u in graph and v in graph:
            controls.append((control, u, v))
    return controls


def farthest_load_buses(graph: nx.Graph, load_kw: dict[str, float], count: int) -> list[str]:
    candidates = [n for n, value in load_kw.items() if value > 0 and n in graph]
    first = max(candidates, key=lambda n: (load_kw[n], n))
    chosen = [first]
    distances = dict(nx.single_source_shortest_path_length(graph, first))
    while len(chosen) < count:
        nxt = max(candidates, key=lambda n: (min(distances.get(n, 10**9), *(nx.shortest_path_length(graph, c, n) for c in chosen)), load_kw[n], n))
        if nxt in chosen:
            remaining = [n for n in candidates if n not in chosen]
            nxt = max(remaining, key=lambda n: min(nx.shortest_path_length(graph, c, n) for c in chosen))
        chosen.append(nxt)
    return chosen


def select_sectionalizers(graph: nx.Graph, load_kw: dict[str, float], count: int, exclude: set[frozenset[str]]) -> list[tuple[str, str]]:
    tree = nx.minimum_spanning_tree(graph)
    root = max(load_kw, key=load_kw.get)
    depth = nx.single_source_shortest_path_length(tree, root)
    candidates = [(u, v) for u, v in tree.edges() if frozenset((u, v)) not in exclude]
    candidates.sort(key=lambda e: (max(depth.get(e[0], 0), depth.get(e[1], 0)), e[0], e[1]))
    positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(i)] for i in positions]


def select_balanced_sectionalizers(graph: nx.Graph, load_kw: dict[str, float], count: int) -> list[tuple[str, str]]:
    """Open cycle-closing edges and partition the spanning tree into load-balanced zones."""
    tree = nx.minimum_spanning_tree(graph)
    selected = [(u, v) for u, v in graph.edges() if not tree.has_edge(u, v)]
    remaining = count - len(selected)
    if remaining < 0:
        raise ValueError("Switch budget is smaller than the graph cycle rank")
    root = max(load_kw, key=load_kw.get)
    parent = {root: None}
    order = [root]
    for u in order:
        for v in tree.neighbors(u):
            if v not in parent:
                parent[v] = u
                order.append(v)
    target = sum(load_kw.values()) / max(remaining + 1, 1)
    accumulated = {n: float(load_kw.get(n, 0.0)) for n in tree}
    cuts: list[tuple[str, str]] = []
    for node in reversed(order[1:]):
        p = parent[node]
        if accumulated[node] >= target and len(cuts) < remaining:
            cuts.append((p, node))
        else:
            accumulated[p] += accumulated[node]
    if len(cuts) < remaining:
        excluded = {frozenset(x) for x in selected + cuts}
        supplements = select_sectionalizers(tree, load_kw, remaining - len(cuts), excluded)
        cuts.extend(supplements)
    return selected + cuts[:remaining]


def component_der_buses(
    graph: nx.Graph,
    switch_pairs: list[tuple[str, str, str]],
    load_kw: dict[str, float],
    count: int,
    capacity_each: float,
) -> list[str]:
    fixed = graph.copy()
    fixed.remove_edges_from((u, v) for _, u, v in switch_pairs)
    ranked = []
    target_initial_load = 0.4 * capacity_each
    for component in nx.connected_components(fixed):
        buses = [n for n in component if load_kw.get(n, 0.0) > 0]
        if not buses:
            continue
        total = sum(load_kw.get(n, 0.0) for n in component)
        representative = max(buses, key=lambda n: (load_kw[n], n))
        ranked.append((abs(total - target_initial_load), -total, representative))
    ranked.sort()
    return [representative for _, _, representative in ranked[:count]]


def assign_agents(
    graph: nx.Graph,
    switches: list[tuple[str, str, str]],
    der_nodes: list[str],
    n_agents: int,
    quotas: list[int] | None = None,
) -> list[int]:
    anchors = der_nodes[:n_agents]
    if len(anchors) < n_agents:
        anchors = anchors + farthest_load_buses(graph, {n: 1.0 for n in graph}, n_agents - len(anchors))
    if quotas is None:
        base, remainder = divmod(len(switches), n_agents)
        quotas = [base + int(k < remainder) for k in range(n_agents)]
    if sum(quotas) != len(switches):
        raise ValueError("Agent switch quotas must sum to the switch count")
    slots = [agent for agent, quota in enumerate(quotas) for _ in range(quota)]
    distances = [nx.single_source_shortest_path_length(graph, anchor) for anchor in anchors]
    costs = np.zeros((len(switches), len(slots)), dtype=np.float64)
    for i, (_, u, v) in enumerate(switches):
        for j, agent in enumerate(slots):
            costs[i, j] = min(distances[agent].get(u, 1e6), distances[agent].get(v, 1e6))
    rows, cols = linear_sum_assignment(costs)
    assignments = [0] * len(switches)
    for row, col in zip(rows, cols):
        assignments[int(row)] = slots[int(col)]
    return assignments


def build_spec(case: str, repo: Path) -> FeederSpec:
    if case == "ieee123":
        master = repo / "external/DSS-Gymnasium/Emergency_Restoration_Rdm_Fault_Training/RandomFaultTrainingCode/IEEE123MasterMultiSW.dss"
        graph, load_kw = extract_graph(master)
        raw_switches = parse_123_switches(master, graph)
        target_switches, n_agents, der_total, der_count = 26, 5, 2600.0, 5
        existing = {frozenset((u, v)) for _, u, v in raw_switches}
        for j, (u, v) in enumerate(select_sectionalizers(graph, load_kw, target_switches - len(raw_switches), existing), 1):
            raw_switches.append((f"assumed{j}", u, v))
        notes = (
            "Topology starts from the public DSS-Gymnasium IEEE123 restoration case at commit 44f2e213e80d89f5c4fe9cb8f90743e8a84e16dc.",
            "Its 23 switch controls are augmented with three deterministic sectionalizers to match the paper's stated 26 switches; the paper does not identify those three devices.",
            "DER locations are deterministic farthest-load buses because paper-specific placements are not released.",
        )
    else:
        master = repo / "external/OpenDSS/Distrib/IEEETestCases/8500-Node/Master.dss"
        graph, load_kw = extract_graph(master)
        target_switches, n_agents, der_total, der_count = 100, 10, 2600.0, 10
        selected = select_balanced_sectionalizers(graph, load_kw, target_switches)
        raw_switches = [(f"assumed{j:03d}", u, v) for j, (u, v) in enumerate(selected, 1)]
        notes = (
            "Topology starts from the official OpenDSS IEEE 8500-node feeder.",
            "One hundred deterministic sectionalizing edges are used because the paper's switch list and modified 25 MW case are not released.",
            "DER locations are deterministic farthest-load buses and total capacity follows the paper's stated 2600 kW.",
        )
    der_nodes = (
        farthest_load_buses(graph, load_kw, der_count)
        if case == "ieee123"
        else component_der_buses(graph, raw_switches, load_kw, der_count, der_total / der_count)
    )
    der_kw = {node: der_total / der_count for node in der_nodes}
    quotas = [10, 5, 3, 3, 5] if case == "ieee123" else [10] * 10
    assignments = assign_agents(graph, raw_switches, der_nodes, n_agents, quotas)
    switch_edges = {frozenset((u, v)) for _, u, v in raw_switches}
    fixed_edges = tuple(sorted((u, v) for u, v in graph.edges() if frozenset((u, v)) not in switch_edges))
    # Priority is assigned independently of load magnitude. Assigning priority by
    # load-magnitude decile (as in an earlier version of this script) implicitly
    # treats "large load" as "critical load", which is backwards for real feeders
    # (hospitals, lift stations, and other critical facilities are frequently
    # small loads). No released dataset labels which nodes on these reconstructed
    # feeders are critical facilities, so priority is instead drawn from a
    # deterministic hash of the node name -- reproducible, but uncorrelated with
    # load size -- with the same 20%/30%/50% -> {3,2,1} class split as before.
    from restoration.environment import stable_rank

    loads_sorted = sorted(load_kw, key=lambda n: stable_rank(f"priority::{n}"))
    priority: dict[str, float] = {}
    for i, node in enumerate(loads_sorted):
        q = i / max(len(loads_sorted), 1)
        priority[node] = 3.0 if q < 0.2 else 2.0 if q < 0.5 else 1.0
    switches = tuple(
        Switch(name=name, u=u, v=v, agent=agent, normally_closed=False)
        for (name, u, v), agent in zip(raw_switches, assignments)
    )
    return FeederSpec(
        name=case,
        nodes=tuple(sorted(graph.nodes())),
        fixed_edges=fixed_edges,
        switches=switches,
        load_kw={k: float(v) for k, v in sorted(load_kw.items())},
        priority=priority,
        der_kw=der_kw,
        n_agents=n_agents,
        source_commit=SOURCE_COMMIT,
        construction_notes=notes,
    )


def write_manifest(specs: list[FeederSpec], output: Path) -> None:
    rows = ["scenario_id,system,split,scenario_type,faulted_switches,der_availability,load_scale"]
    for spec in specs:
        single_ids = [f"{spec.name}_single_{i:03d}" for i in range(len(spec.switches))]
        split = deterministic_split(single_ids)
        for i, scenario_id in enumerate(single_ids):
            rows.append(f"{scenario_id},{spec.name},{split[scenario_id]},single_fault,{i},1.0,1.0")
        test_indices = [i for i, sid in enumerate(single_ids) if split[sid] == "test"]
        for j in range(min(10, max(1, len(test_indices) - 1))):
            pair = sorted({test_indices[j % len(test_indices)], test_indices[(j + 1) % len(test_indices)]})
            rows.append(f"{spec.name}_multi_{j:03d},{spec.name},test,multiple_fault,{'|'.join(map(str, pair))},1.0,1.0")
        for j, availability in enumerate((0.9, 0.8, 0.7)):
            fault = test_indices[j % len(test_indices)]
            rows.append(f"{spec.name}_der_{j:03d},{spec.name},test,der_variation,{fault},{availability},1.0")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = [build_spec("ieee123", repo), build_spec("ieee8500", repo)]
    for spec in specs:
        spec.save(output / f"{spec.name}_feeder.json")
    write_manifest(specs, repo / "DATA_SPLIT_MANIFEST.csv")
    summary = {
        spec.name: {
            "nodes": len(spec.nodes),
            "fixed_edges": len(spec.fixed_edges),
            "switches": len(spec.switches),
            "agents": spec.n_agents,
            "load_kw": spec.total_load_kw,
            "der_kw": spec.total_der_kw,
        }
        for spec in specs
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
