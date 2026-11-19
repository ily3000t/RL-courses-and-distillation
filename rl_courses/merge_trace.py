"""Optional TraCI observations for replay diagnostics and reference profiles."""

from traci.constants import LCA_BLOCKED, LCA_OVERLAPPING


def before_tick(connection, ego, merge_end_x):
    vehicles = connection.vehicle
    lane = vehicles.getLaneID(ego)
    x = vehicles.getPosition(ego)[0]
    speed = vehicles.getSpeed(ego)
    row = {
        "time_s": connection.simulation.getTime(), "lane": lane, "speed_mps": speed,
        "distance_to_merge_end_m": merge_end_x - x,
        "front": None, "rear": None, "gap_proxy_open": False,
    }
    if lane != "merge_0":
        return row
    remaining = connection.lane.getLength(lane) - vehicles.getLanePosition(ego)
    for key, mode in (("front", 2), ("rear", 0)):
        neighbors = vehicles.getNeighbors(ego, mode)
        if not neighbors:
            continue
        neighbor, distance = neighbors[0]
        other_speed = vehicles.getSpeed(neighbor)
        follower, leader = (ego, neighbor) if key == "front" else (neighbor, ego)
        follower_speed, leader_speed = (speed, other_speed) if key == "front" else (other_speed, speed)
        secure = vehicles.getSecureGap(
            follower, follower_speed, leader_speed, vehicles.getDecel(leader), leader
        )
        row[key] = {"gap_m": distance, "secure_gap_m": secure, "relative_speed_mps": other_speed - speed}
    row["gap_proxy_open"] = remaining > 3 and all(
        neighbor is None or neighbor["gap_m"] >= neighbor["secure_gap_m"]
        for neighbor in (row["front"], row["rear"])
    )
    return row


def after_tick(connection, ego, row, requested, acceleration, intent):
    lane_after = None
    blocked = None
    if ego in connection.vehicle.getIDList():
        lane_after = connection.vehicle.getLaneID(ego)
        if requested and lane_after == "merge_0":
            _, state_after_traci = connection.vehicle.getLaneChangeState(ego, 1)
            blocked = bool(state_after_traci & (LCA_BLOCKED | LCA_OVERLAPPING))
    return {
        **row, "lane_after": lane_after, "requested": requested,
        "blocked_by_sumo": blocked, "action_acceleration": float(acceleration),
        "action_merge_intent": float(intent),
    }
