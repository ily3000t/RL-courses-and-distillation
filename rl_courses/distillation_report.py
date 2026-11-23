"""Report matched student and teacher closed-loop results and deployment costs."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev

from .distillation import STUDENT_METHODS
from .evaluation import summarize
from .records import read_json
from .reporting import METRICS


def _read_completed(directory):
    if read_json(directory / "status.json")["state"] != "completed":
        raise ValueError(f"Run is incomplete: {directory}")
    return read_json(directory / "manifest.json")


def _read_evaluation(directory, split):
    meta = _read_completed(directory)
    checkpoint = Path(meta["checkpoint"])
    train = _read_completed(checkpoint.parent)
    if (meta["split"] != split or checkpoint.name != "best.zip" or meta["checkpoint_kind"] != "best"
            or sha256(checkpoint.read_bytes()).hexdigest() != meta["checkpoint_sha256"]
            or meta["training_git_sha"] != train["git_sha"] or meta["seed"] != train["seed"]
            or meta["config"] != train["config"]):
        raise ValueError(f"Evaluation provenance does not match its training run: {directory}")
    rows = read_json(directory / "episodes.json")
    expected = [(row["scene_id"], row["seed"], row["difficulty"]) for row in meta["evaluation_catalog"]]
    observed = [(row["scene_id"], row["scene_seed"], row["difficulty"]) for row in rows]
    if observed != expected:
        raise ValueError(f"Evaluation rows do not match the scenario catalog: {directory}")
    groups = defaultdict(list)
    for row in rows:
        groups[row["difficulty"]].append(row)
    deployment = read_json(directory / "deployment.json")
    artifact = directory / "actor.pt"
    if (artifact.stat().st_size != deployment["file_bytes"]
            or sha256(artifact.read_bytes()).hexdigest() != deployment["actor_sha256"]):
        raise ValueError(f"Exported actor differs from the deployment record: {directory}")
    return meta, train, {
        "all": summarize(rows), **{difficulty: summarize(group) for difficulty, group in groups.items()}
    }, deployment


def summarize_distillation(root, split):
    data = root / "collection"
    collection = _read_completed(data)
    collection_summary = read_json(data / "collection_summary.json")
    teacher = collection["teacher"]
    tm, _, metrics, deployment = _read_evaluation(root / "teacher" / split, split)
    if (tm["checkpoint_sha256"] != teacher["checkpoint_sha256"]
            or tm["training_git_sha"] != teacher["git_sha"] or tm["seed"] != teacher["seed"]):
        raise ValueError("The evaluated teacher differs from the collection teacher.")
    teacher_result = {"checkpoint": teacher, "evaluation": metrics, "deployment": deployment}
    eval_protocol = (tm["evaluation_catalog"], tm["versions"], tm["git_sha"], tm["stress_holdout_seed"],
                     tm["config"]["environment"], tm["observation_fields"])
    deployment_fields = ("format", "device", "cpu_model", "cpu_threads", "batch_size", "timed_calls")
    benchmark = tuple(deployment[key] for key in deployment_fields)
    catalog = read_json(data / "scenes.json")
    student_protocol = None
    students = {}
    for method in STUDENT_METHODS:
        runs = {}
        for path in sorted((root / method).glob(f"seed-*/{split}")):
            meta, train, evaluation, actor = _read_evaluation(path, split)
            config = train["config"].copy()
            if config.pop("method") != method or read_json(Path(meta["checkpoint"]).parent / "scenes.json") != catalog:
                raise ValueError(f"Student method or training catalog differs: {path}")
            protocol = (config, train["versions"])
            if student_protocol is None:
                student_protocol = protocol
            if protocol != student_protocol:
                raise ValueError(f"Student settings differ across methods or seeds: {path}")
            if eval_protocol != (meta["evaluation_catalog"], meta["versions"], meta["git_sha"],
                                 meta["stress_holdout_seed"], meta["config"]["environment"],
                                 meta["observation_fields"]):
                raise ValueError(f"Student and teacher evaluation protocols differ: {path}")
            if tuple(actor[key] for key in deployment_fields) != benchmark:
                raise ValueError(f"Student and teacher actor benchmarks differ: {path}")
            if method != "scratch":
                source = train["teacher_data"]
                if (source["dataset_sha256"] != collection_summary["dataset_sha256"]
                        or source["teacher"] != teacher or source["collection_seed"] != collection["seed"]):
                    raise ValueError(f"Student was trained from a different teacher dataset: {path}")
            seed = meta["seed"]
            if seed in runs or path.parent.name != f"seed-{seed}":
                raise ValueError(f"Duplicate or misplaced seed: {path}")
            runs[seed] = {"seed": seed, "training_git_sha": train["git_sha"],
                          "checkpoint": meta["checkpoint"], "evaluation": evaluation,
                          "deployment": actor, "training": read_json(Path(meta["checkpoint"]).parent / "training_summary.json")}
        students[method] = runs
    seeds = sorted(students["scratch"])
    if not seeds or any(set(runs) != set(seeds) for runs in students.values()):
        raise ValueError("All three student methods must have the same nonempty seed set.")
    if len({run["deployment"]["actor_parameters"] for runs in students.values() for run in runs.values()}) != 1:
        raise ValueError("Student deployment actor sizes differ.")

    aggregates = {}
    for method, runs in students.items():
        hard_metrics = {}
        for metric in METRICS:
            values = [runs[seed]["evaluation"]["hard"][metric] for seed in seeds]
            hard_metrics[metric] = {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else None}
        teacher_success = metrics["hard"]["success_rate"]
        success = hard_metrics["success_rate"]["mean"]
        aggregates[method] = {
            "hard_evaluation": hard_metrics, "teacher_hard_success_delta": success - teacher_success,
            "teacher_hard_success_retention": success / teacher_success if teacher_success else None,
            "teacher_to_student_parameter_ratio": deployment["actor_parameters"] / runs[seeds[0]]["deployment"]["actor_parameters"],
            "runs": [runs[seed] for seed in seeds],
        }
    return {"split": split, "stress_holdout_seed": tm["stress_holdout_seed"],
            "evaluation_git_sha": tm["git_sha"], "evaluation_catalog": tm["evaluation_catalog"],
            "student_seeds": seeds, "teacher": teacher_result, "collection": collection_summary,
            "student_config": student_protocol[0], "students": aggregates,
            "soft_minus_bc_hard": {
                metric: [students["soft"][seed]["evaluation"]["hard"][metric]
                         - students["bc"][seed]["evaluation"]["hard"][metric] for seed in seeds]
                for metric in METRICS
            }}
