"""Deterministic closed-loop evaluation on an explicit list of scenarios."""

from collections import Counter

from .catalog import group_by_difficulty


OUTCOMES = ("success", "collision", "timeout", "missed_merge")


def is_pre_merge_timeout(episode):
    return episode["outcome"] == "timeout" and not episode["merged"]


def is_pre_merge_failure(episode):
    return episode["outcome"] == "missed_merge" or is_pre_merge_timeout(episode)


def evaluate(model, env, scenes):
    episodes = []
    for scene in scenes:
        observation, _ = env.reset(seed=scene.seed, options={"scene": scene})
        total_reward = 0.0
        decisions = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            decisions += 1
            if terminated or truncated:
                episodes.append({**info, "return": total_reward, "decisions": decisions})
                break
    summary = {"all": summarize(episodes)}
    for difficulty, group in group_by_difficulty(scenes).items():
        ids = {scene.scene_id for scene in group}
        summary[difficulty] = summarize([row for row in episodes if row["scene_id"] in ids])
    return summary, episodes


def summarize(episodes):
    counts = Counter(row["outcome"] for row in episodes)
    count = len(episodes)
    pre_merge_timeouts = sum(map(is_pre_merge_timeout, episodes))
    return {
        "episodes": count,
        "mean_return": sum(row["return"] for row in episodes) / count,
        "pre_merge_timeout_rate": pre_merge_timeouts / count,
        "pre_merge_failure_rate": (pre_merge_timeouts + counts["missed_merge"]) / count,
        **{f"{outcome}_rate": counts[outcome] / count for outcome in OUTCOMES},
    }


def validation_key(summary):
    hard = summary["hard"]
    return (
        hard["success_rate"], -hard["collision_rate"], -hard["timeout_rate"], hard["mean_return"]
    )
