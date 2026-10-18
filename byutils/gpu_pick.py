"""Rank partition/QOS options for a single-node GPU job from live Slurm state.

    python -m byutils.gpu_pick --hours 3 --vram-gb 130 [--exclude-qos dw87]
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

CpuArch = Literal["x86_64", "aarch64"]


@dataclass(frozen=True)
class GpuSpec:
    memory_gb: float
    min_cuda: tuple[int, int]
    max_cuda: tuple[int, int] | None = None


GPU_SPEC_BY_TYPE: dict[str, GpuSpec] = {
    "p100": GpuSpec(memory_gb=16, min_cuda=(8, 0), max_cuda=(12, 9)),
    "l40s": GpuSpec(memory_gb=48, min_cuda=(11, 8)),
    "a100": GpuSpec(memory_gb=80, min_cuda=(11, 0)),
    "h100": GpuSpec(memory_gb=80, min_cuda=(11, 8)),
    "gh200": GpuSpec(memory_gb=96, min_cuda=(12, 0)),
    "h200": GpuSpec(memory_gb=141, min_cuda=(12, 0)),
    "b200": GpuSpec(memory_gb=180, min_cuda=(12, 8)),
}
CPU_ARCH_BY_PARTITION: dict[str, CpuArch] = {"mgh": "aarch64"}
STANDBY_GPU_QOS = "gstandby"
COMPETING_PENDING_REASONS = frozenset({"Priority", "Resources"})
USABLE_NODE_STATES = frozenset({"idle", "mix", "alloc", "mix-", "alloc-", "idle-", "comp"})


@dataclass(frozen=True)
class Qos:
    name: str
    priority: int
    usage_factor: float
    max_wall_hours: float | None
    preempted_qos_names: frozenset[str]


@dataclass(frozen=True)
class Partition:
    name: str
    allowed_qos_names: frozenset[str]
    max_time_hours: float | None
    gpu_billing_weight: float
    cpu_billing_weight: float
    mem_billing_weight_per_gb: float


@dataclass(frozen=True)
class NodeResources:
    gpus: int
    cpus: int
    mem_gb: float


@dataclass(frozen=True)
class GpuNode:
    name: str
    partition_name: str
    gpu_type: str
    total_gpus: int
    free: NodeResources


@dataclass(frozen=True)
class PendingJob:
    partition_names: frozenset[str]
    qos_name: str
    planned_start: datetime | None
    reserved_node_names: frozenset[str]
    requested_gpu_type: str | None
    requested: NodeResources


@dataclass(frozen=True)
class JobRequest:
    vram_gb: float
    hours: float
    cpus: int
    mem_gb: float
    requeue_ok: bool
    fast_turnaround: bool
    cpu_arch: CpuArch
    cuda: tuple[int, int]
    excluded_gpu_types: frozenset[str]
    excluded_qos_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Candidate:
    partition_name: str
    gpu_type: str
    qos: Qos
    submit_qos_name: str
    gpu_count: int
    max_wall_hours: float
    idle_gpus: int
    preemptable_gpus: int
    fitting_node_count: int
    open_node_count: int
    startable_now: bool
    pending_jobs_ahead: int
    billing_per_hour: float
    total_billing: float


def run_command(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def parse_slurm_duration_hours(duration: str) -> float | None:
    if duration in ("", "UNLIMITED", "Partition_Limit", "INFINITE"):
        return None
    days, _, clock = duration.rpartition("-")
    clock_parts = [int(part) for part in clock.split(":")]
    hours, minutes, seconds = [0] * (3 - len(clock_parts)) + clock_parts
    return (int(days) if days else 0) * 24 + hours + minutes / 60 + seconds / 3600


def parse_billing_weights(weights_text: str) -> tuple[float, float, float]:
    weight_by_tres = dict(item.split("=", 1) for item in weights_text.split(",") if "=" in item)
    mem_weight_text = weight_by_tres.get("Mem", "0")
    mem_weight_per_gb = float(mem_weight_text[:-1]) if mem_weight_text.endswith("G") else float(mem_weight_text) * 1024
    return float(weight_by_tres.get("GRES/gpu", "0")), float(weight_by_tres.get("CPU", "0")), mem_weight_per_gb


def load_user_qos_names() -> frozenset[str]:
    output = run_command(["sacctmgr", "-nP", "show", "assoc", "user=" + run_command(["whoami"]).strip(), "format=QOS"])
    return frozenset(name for line in output.split() for name in line.split(",") if name)


def load_qos_by_name() -> dict[str, Qos]:
    output = run_command(["sacctmgr", "-nP", "show", "qos", "format=Name,Priority,UsageFactor,MaxWall,Preempt"])
    rows = [line.split("|") for line in output.splitlines() if line]
    return {
        name: Qos(
            name=name,
            priority=int(priority or 0),
            usage_factor=float(usage_factor),
            max_wall_hours=parse_slurm_duration_hours(max_wall),
            preempted_qos_names=frozenset(filter(None, preempt.split(","))),
        )
        for name, priority, usage_factor, max_wall, preempt in rows
    }


def load_partitions() -> list[Partition]:
    output = run_command(["scontrol", "show", "partition", "-o"])
    field_maps = [dict(re.findall(r"(\w+)=(\S*)", line)) for line in output.splitlines() if line]
    return [
        Partition(
            name=fields["PartitionName"],
            allowed_qos_names=frozenset(fields.get("AllowQos", "").split(",")),
            max_time_hours=parse_slurm_duration_hours(fields.get("MaxTime", "")),
            gpu_billing_weight=gpu_weight,
            cpu_billing_weight=cpu_weight,
            mem_billing_weight_per_gb=mem_weight,
        )
        for fields in field_maps
        for gpu_weight, cpu_weight, mem_weight in [parse_billing_weights(fields.get("TRESBillingWeights", ""))]
        if gpu_weight > 0
    ]


def parse_gres_gpu(gres_text: str) -> tuple[str, int] | None:
    match = re.search(r"gpu:([a-z0-9_]+):(\d+)", gres_text)
    return (match.group(1), int(match.group(2))) if match else None


def load_gpu_nodes(partition_names: list[str]) -> list[GpuNode]:
    output = run_command([
        "sinfo", "-h", "-N", "-p", ",".join(partition_names),
        "-O", "Partition:30,NodeList:40,StateCompact:14,CPUsState:24,AllocMem:14,Memory:14,Gres:60,GresUsed:80",
    ])
    nodes: list[GpuNode] = []
    for line in output.splitlines():
        partition_name, node_name, state, cpu_states, alloc_mem_mb, total_mem_mb, gres_text, gres_used_text = line.split()[:8]
        gpu_total = parse_gres_gpu(gres_text)
        gpu_used = parse_gres_gpu(gres_used_text)
        if gpu_total is None or state not in USABLE_NODE_STATES:
            continue
        _, idle_cpus, _, _ = cpu_states.split("/")
        nodes.append(GpuNode(
            name=node_name, partition_name=partition_name.rstrip("*"), gpu_type=gpu_total[0], total_gpus=gpu_total[1],
            free=NodeResources(
                gpus=gpu_total[1] - (gpu_used[1] if gpu_used else 0),
                cpus=int(idle_cpus),
                mem_gb=(int(total_mem_mb) - int(alloc_mem_mb)) / 1024,
            ),
        ))
    return nodes


def parse_tres_mem_gb(tres_text: str) -> float:
    match = re.search(r"mem=([\d.]+)([KMGT])", tres_text)
    if not match:
        return 0
    gb_per_unit_by_suffix = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}
    return float(match.group(1)) * gb_per_unit_by_suffix[match.group(2)]


def add_resources(first: NodeResources, second: NodeResources) -> NodeResources:
    return NodeResources(gpus=first.gpus + second.gpus, cpus=first.cpus + second.cpus, mem_gb=first.mem_gb + second.mem_gb)


def expand_hostlist(hostlist: str) -> list[str]:
    return run_command(["scontrol", "show", "hostnames", hostlist]).split()


def load_standby_resources_by_node() -> dict[str, NodeResources]:
    output = run_command(["squeue", "-h", "-t", "R", "-q", STANDBY_GPU_QOS, "-O", "NumNodes:8,NodeList:200,tres-alloc:300"])
    standby_resources_by_node: dict[str, NodeResources] = {}
    for line in output.splitlines():
        node_count_text, hostlist, tres_text = line.split()[:3]
        gpu_match = re.search(r"gres/gpu=(\d+)", tres_text)
        cpu_match = re.search(r"cpu=(\d+)", tres_text)
        if not gpu_match or not cpu_match:
            continue
        node_count = int(node_count_text)
        per_node = NodeResources(
            gpus=int(gpu_match.group(1)) // node_count,
            cpus=int(cpu_match.group(1)) // node_count,
            mem_gb=parse_tres_mem_gb(tres_text) / node_count,
        )
        for node_name in expand_hostlist(hostlist):
            standby_resources_by_node[node_name] = add_resources(
                standby_resources_by_node.get(node_name, NodeResources(gpus=0, cpus=0, mem_gb=0)), per_node
            )
    return standby_resources_by_node


def parse_planned_start(start_text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(start_text)
    except ValueError:
        return None


def parse_requested_gpus(tres_per_job: str, tres_per_node: str) -> tuple[str | None, int]:
    gpu_request = next((tres for tres in (tres_per_job, tres_per_node) if "gpu" in tres), None)
    if gpu_request is None:
        return None, 0
    request_parts = gpu_request.split(":")
    return (request_parts[1] if len(request_parts) == 3 else None), int(request_parts[-1])


def load_competing_pending_jobs() -> list[PendingJob]:
    output = run_command([
        "squeue", "-h", "-t", "PD",
        "-O", "Partition:200,QOS:30,StartTime:24,SchedNodes:200,Reason:60,MinCpus:10,MinMemory:14,tres-per-job:60,tres-per-node:60",
    ])
    rows = [line.split() for line in output.splitlines() if line.strip()]
    eligible_rows = [row for row in rows if len(row) >= 9 and row[4] in COMPETING_PENDING_REASONS]
    node_names_by_hostlist = {
        hostlist: frozenset(expand_hostlist(hostlist))
        for hostlist in {row[3] for row in eligible_rows} if hostlist != "(null)"
    }
    return [
        PendingJob(
            partition_names=frozenset(row[0].split(",")),
            qos_name=row[1],
            planned_start=parse_planned_start(row[2]),
            reserved_node_names=node_names_by_hostlist.get(row[3], frozenset()),
            requested_gpu_type=requested_gpu_type,
            requested=NodeResources(gpus=requested_gpu_count, cpus=int(row[5]), mem_gb=parse_tres_mem_gb("mem=" + row[6])),
        )
        for row in eligible_rows
        for requested_gpu_type, requested_gpu_count in [parse_requested_gpus(row[7], row[8])]
    ]


def load_backfill_resolution_hours() -> float:
    config = run_command(["scontrol", "show", "config"])
    match = re.search(r"bf_resolution=(\d+)", config)
    return int(match.group(1)) / 3600 if match else 60 / 3600


def gpu_job_qos_names(user_qos_names: frozenset[str]) -> dict[str, str]:
    submit_name_by_effective_qos = {name: name for name in user_qos_names if name not in ("standby", STANDBY_GPU_QOS)}
    if "standby" in user_qos_names:
        submit_name_by_effective_qos[STANDBY_GPU_QOS] = "standby"
    return submit_name_by_effective_qos


def billing_per_hour(partition: Partition, gpu_count: int, request: JobRequest) -> float:
    return max(
        partition.gpu_billing_weight * gpu_count,
        partition.cpu_billing_weight * request.cpus,
        partition.mem_billing_weight_per_gb * request.mem_gb,
    )


def combined_max_wall_hours(qos: Qos, partition: Partition) -> float:
    limits = [limit for limit in (qos.max_wall_hours, partition.max_time_hours) if limit is not None]
    return min(limits) if limits else math.inf


def is_gpu_allowed(gpu_type: str, partition_name: str, request: JobRequest) -> bool:
    spec = GPU_SPEC_BY_TYPE.get(gpu_type)
    return (
        spec is not None
        and gpu_type not in request.excluded_gpu_types
        and request.cuda >= spec.min_cuda
        and (spec.max_cuda is None or request.cuda <= spec.max_cuda)
        and CPU_ARCH_BY_PARTITION.get(partition_name, "x86_64") == request.cpu_arch
    )


def fits_on_node(available: NodeResources, gpu_count: int, request: JobRequest) -> bool:
    return available.gpus >= gpu_count and available.cpus >= request.cpus and available.mem_gb >= request.mem_gb


def is_claimed_by_higher_priority_job(
    node: GpuNode,
    qos: Qos,
    pending_jobs: list[PendingJob],
    qos_by_name: dict[str, Qos],
    standby_resources_by_node: dict[str, NodeResources],
) -> bool:
    no_resources = NodeResources(gpus=0, cpus=0, mem_gb=0)
    for job in pending_jobs:
        job_qos = qos_by_name.get(job.qos_name)
        if (
            job_qos is None
            or job_qos.priority < qos.priority
            or node.partition_name not in job.partition_names
            or job.requested_gpu_type not in (None, node.gpu_type)
        ):
            continue
        preemptable = standby_resources_by_node.get(node.name, no_resources) if STANDBY_GPU_QOS in job_qos.preempted_qos_names else no_resources
        available = add_resources(node.free, preemptable)
        if available.gpus >= job.requested.gpus and available.cpus >= job.requested.cpus and available.mem_gb >= job.requested.mem_gb:
            return True
    return False


def hours_until_reserved(
    node_name: str, qos: Qos, pending_jobs: list[PendingJob], qos_by_name: dict[str, Qos], now: datetime
) -> float:
    reservation_starts = [
        job.planned_start for job in pending_jobs
        if job.planned_start is not None
        and node_name in job.reserved_node_names
        and job.qos_name in qos_by_name
        and qos_by_name[job.qos_name].priority >= qos.priority
    ]
    return (min(reservation_starts) - now).total_seconds() / 3600 if reservation_starts else math.inf


def count_pending_jobs_ahead(partition_name: str, qos: Qos, pending_jobs: list[PendingJob], qos_by_name: dict[str, Qos]) -> int:
    return sum(
        1 for job in pending_jobs
        if partition_name in job.partition_names
        and job.qos_name in qos_by_name
        and qos_by_name[job.qos_name].priority >= qos.priority
    )


def build_candidates(
    request: JobRequest,
    partitions: list[Partition],
    gpu_nodes: list[GpuNode],
    standby_resources_by_node: dict[str, NodeResources],
    pending_jobs: list[PendingJob],
    qos_by_name: dict[str, Qos],
    submit_name_by_effective_qos: dict[str, str],
    backfill_resolution_hours: float,
    now: datetime,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for partition in partitions:
        partition_nodes = [node for node in gpu_nodes if node.partition_name == partition.name]
        if not partition_nodes:
            continue
        gpu_type = partition_nodes[0].gpu_type
        if not is_gpu_allowed(gpu_type, partition.name, request):
            continue
        gpu_count = max(1, math.ceil(request.vram_gb / GPU_SPEC_BY_TYPE[gpu_type].memory_gb))
        if gpu_count > max(node.total_gpus for node in partition_nodes):
            continue
        for effective_qos_name, submit_qos_name in submit_name_by_effective_qos.items():
            qos = qos_by_name.get(effective_qos_name)
            if qos is None or effective_qos_name not in partition.allowed_qos_names:
                continue
            if submit_qos_name in request.excluded_qos_names:
                continue
            if effective_qos_name == STANDBY_GPU_QOS and not request.requeue_ok:
                continue
            max_wall_hours = combined_max_wall_hours(qos, partition)
            if request.hours > max_wall_hours:
                continue
            can_preempt_standby = STANDBY_GPU_QOS in qos.preempted_qos_names
            no_resources = NodeResources(gpus=0, cpus=0, mem_gb=0)
            preemptable_by_node = {
                node.name: standby_resources_by_node.get(node.name, no_resources) if can_preempt_standby else no_resources
                for node in partition_nodes
            }
            fitting_nodes = [
                node for node in partition_nodes
                if fits_on_node(add_resources(node.free, preemptable_by_node[node.name]), gpu_count, request)
            ]
            open_node_count = sum(
                not is_claimed_by_higher_priority_job(node, qos, pending_jobs, qos_by_name, standby_resources_by_node)
                and hours_until_reserved(node.name, qos, pending_jobs, qos_by_name, now) >= request.hours + backfill_resolution_hours
                for node in fitting_nodes
            )
            pending_jobs_ahead = count_pending_jobs_ahead(partition.name, qos, pending_jobs, qos_by_name)
            hourly_billing = billing_per_hour(partition, gpu_count, request) * qos.usage_factor
            candidates.append(Candidate(
                partition_name=partition.name,
                gpu_type=gpu_type,
                qos=qos,
                submit_qos_name=submit_qos_name,
                gpu_count=gpu_count,
                max_wall_hours=max_wall_hours,
                idle_gpus=sum(node.free.gpus for node in partition_nodes),
                preemptable_gpus=sum(resources.gpus for resources in preemptable_by_node.values()),
                fitting_node_count=len(fitting_nodes),
                open_node_count=open_node_count,
                startable_now=open_node_count > 0,
                pending_jobs_ahead=pending_jobs_ahead,
                billing_per_hour=hourly_billing,
                total_billing=hourly_billing * request.hours,
            ))
    return candidates


def qos_preference_rank(candidate: Candidate, fast_turnaround: bool) -> int:
    if candidate.qos.name == "cs":
        return 0
    if candidate.qos.name == STANDBY_GPU_QOS:
        return 3 if fast_turnaround and candidate.pending_jobs_ahead > 0 else 1
    return 2


def rank_candidates(candidates: list[Candidate], fast_turnaround: bool) -> list[Candidate]:
    return sorted(candidates, key=lambda candidate: (
        not candidate.startable_now,
        candidate.gpu_count,
        qos_preference_rank(candidate, fast_turnaround),
        candidate.total_billing,
        candidate.pending_jobs_ahead,
    ))


def format_hours_as_slurm_time(hours: float) -> str:
    total_minutes = math.ceil(hours * 60)
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    clock = f"{remaining_minutes // 60:02d}:{remaining_minutes % 60:02d}:00"
    return f"{days}-{clock}" if days else clock


def suggested_sbatch_flags(ranked_candidates: list[Candidate], request: JobRequest) -> list[str]:
    best = ranked_candidates[0]
    same_submission = [
        candidate.partition_name for candidate in ranked_candidates
        if candidate.submit_qos_name == best.submit_qos_name
        and candidate.gpu_count == best.gpu_count
        and candidate.startable_now == best.startable_now
    ]
    gpu_value = f"{best.gpu_type}:{best.gpu_count}" if len(same_submission) == 1 else str(best.gpu_count)
    return [
        "-p", ",".join(same_submission), "--qos", best.submit_qos_name, "--gpus", gpu_value,
        "--time", format_hours_as_slurm_time(request.hours),
    ]


def suggest_sbatch_command(ranked_candidates: list[Candidate], request: JobRequest) -> str:
    return " ".join(["sbatch", *suggested_sbatch_flags(ranked_candidates, request)])


def format_candidate_table(ranked_candidates: list[Candidate]) -> str:
    header = f"{'#':>2}  {'partition':<9} {'gpu':<6} {'n':>2}  {'--qos':<9} {'prio':>4}  {'now?':<4} {'idle':>4} {'preempt':>7} {'fit/open':>8} {'jobs ahead':>10}  {'cost/hr':>8} {'cost':>9}  {'max wall':>8}"
    rows = [
        f"{index:>2}  {c.partition_name:<9} {c.gpu_type:<6} {c.gpu_count:>2}  {c.submit_qos_name:<9} {c.qos.priority:>4}  "
        f"{'yes' if c.startable_now else 'no':<4} {c.idle_gpus:>4} {c.preemptable_gpus:>7} {f'{c.fitting_node_count}/{c.open_node_count}':>8} {c.pending_jobs_ahead:>10}  "
        f"{c.billing_per_hour:>8.1f} {c.total_billing:>9.0f}  {format_hours_as_slurm_time(c.max_wall_hours) if math.isfinite(c.max_wall_hours) else '-':>8}"
        for index, c in enumerate(ranked_candidates, start=1)
    ]
    return "\n".join([header, *rows])


def parse_cuda_version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")
    return int(major), int(minor)


def parse_request() -> tuple[JobRequest, int]:
    parser = argparse.ArgumentParser(description="Rank partition/QOS options for a single-node GPU job, using live Slurm state.")
    parser.add_argument("--hours", type=float, required=True, help="Wall time the job needs")
    parser.add_argument("--vram-gb", type=float, default=0, help="Total GPU memory needed; sets the GPU count per GPU type")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem-gb", type=float, default=0, help="Host memory; counts toward billing under MAX_TRES")
    parser.add_argument("--requeue-ok", action=argparse.BooleanOptionalAction, default=True,
                        help="Job survives preemption+requeue (allows --qos standby)")
    parser.add_argument("--fast", action="store_true", help="Rank contended standby options below full-price ones")
    parser.add_argument("--arch", choices=["x86_64", "aarch64"], default="x86_64")
    parser.add_argument("--cuda", type=parse_cuda_version, default=(12, 6), help="CUDA version the env is built for, e.g. 12.6")
    parser.add_argument("--exclude-gpu", action="append", default=[], help="GPU type to skip, e.g. p100 (repeatable)")
    parser.add_argument("--exclude-qos", action="append", default=[], help="QOS to skip, e.g. dw87 (repeatable)")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    return JobRequest(
        vram_gb=args.vram_gb, hours=args.hours, cpus=args.cpus, mem_gb=args.mem_gb,
        requeue_ok=args.requeue_ok, fast_turnaround=args.fast, cpu_arch=args.arch, cuda=args.cuda,
        excluded_gpu_types=frozenset(gpu.lower() for gpu in args.exclude_gpu),
        excluded_qos_names=frozenset(args.exclude_qos),
    ), args.top


def ranked_gpu_candidates(request: JobRequest) -> list[Candidate]:
    """Every partition/QOS option for the request, best first; empty if none satisfies it."""
    qos_by_name = load_qos_by_name()
    partitions = load_partitions()
    candidates = build_candidates(
        request=request,
        partitions=partitions,
        gpu_nodes=load_gpu_nodes([partition.name for partition in partitions]),
        standby_resources_by_node=load_standby_resources_by_node(),
        pending_jobs=load_competing_pending_jobs(),
        qos_by_name=qos_by_name,
        submit_name_by_effective_qos=gpu_job_qos_names(load_user_qos_names()),
        backfill_resolution_hours=load_backfill_resolution_hours(),
        now=datetime.now(),
    )
    return rank_candidates(candidates, request.fast_turnaround)


def main() -> None:
    request, top_count = parse_request()
    ranked_candidates = ranked_gpu_candidates(request)
    if not ranked_candidates:
        print("No partition/QOS satisfies these constraints.")
        return
    print(format_candidate_table(ranked_candidates[:top_count]))
    print()
    print(suggest_sbatch_command(ranked_candidates, request))


if __name__ == "__main__":
    main()
