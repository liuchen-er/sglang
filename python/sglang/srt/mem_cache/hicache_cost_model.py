import json

class HiCacheCostModel:
    def __init__(self, profile_path: str, safety_margin_ms: float = 1.0):
        with open(profile_path, encoding="utf-8") as f:
            self.profile = json.load(f)

        if self.profile.get("version") != 2:
            raise ValueError(
                f"HiCache cost profile version must be 2, got {self.profile.get('version')}"
            )

        self.safety_margin_ms = safety_margin_ms
        self.running_bs_buckets = sorted(self.profile["running_bs_buckets"])
        self.curves = {"restore": {}, "recompute": {}}

        for action in ("restore", "recompute"):
            for entry in self.profile["entries"][action]:
                bs = int(entry["running_bs"])
                self.curves[action].setdefault(bs, []).append(entry)

            for bs in self.curves[action]:
                self.curves[action][bs].sort(key=lambda x: x["host_hit"])

    def _nearest_running_bs(self, running_bs: int) -> int:
        return min(self.running_bs_buckets, key=lambda x: abs(x - running_bs))

    @staticmethod
    def _interp(curve, host_hit):
        if not curve:
            return None

        min_hit = curve[0]["host_hit"]
        max_hit = curve[-1]["host_hit"]

        # Do not extrapolate outside the profiled host-hit range.
        if host_hit < min_hit or host_hit > max_hit:
            return None

        for point in curve:
            if point["host_hit"] == host_hit:
                return point["latency_ms"]

        # 边界检查
        for left, right in zip(curve, curve[1:]):
            if left["host_hit"] < host_hit < right["host_hit"]:
                x0, x1 = left["host_hit"], right["host_hit"]
                y0, y1 = left["latency_ms"], right["latency_ms"]
                ratio = (host_hit - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)

        return None

    def estimate(self, host_hit: int, running_bs: int):
        bs = self._nearest_running_bs(running_bs)

        restore_curve = self.curves["restore"].get(bs)
        recompute_curve = self.curves["recompute"].get(bs)

        restore_ms = self._interp(restore_curve, host_hit)
        recompute_ms = self._interp(recompute_curve, host_hit)

        return restore_ms, recompute_ms

    def decide(self, host_hit: int, running_bs: int):
        restore_ms, recompute_ms = self.estimate(host_hit, running_bs)

        # OOD / incomplete profile: fall back to native safe behavior.
        if restore_ms is None or recompute_ms is None:
            return True, restore_ms, recompute_ms

        should_restore = restore_ms + self.safety_margin_ms < recompute_ms
        return should_restore, restore_ms, recompute_ms
