import json


class HiCacheCostModel:
    def __init__(self, profile_path: str, safety_margin_ms: float = 2.0):
        with open(profile_path, encoding="utf-8") as f:
            self.profile = json.load(f)

        self.version = self.profile.get("version")
        if self.version not in (2, 3):
            raise ValueError(
                f"HiCache cost profile version must be 2 or 3, got {self.version}"
            )

        self.safety_margin_ms = safety_margin_ms

        # L2 model: preserve V2 behavior.
        self.running_bs_buckets = sorted(self.profile["running_bs_buckets"])
        self.curves = {"restore": {}, "recompute": {}}

        for action in ("restore", "recompute"):
            for entry in self.profile["entries"][action]:
                bs = int(entry["running_bs"])
                self.curves[action].setdefault(bs, []).append(entry)

            for bs in self.curves[action]:
                self.curves[action][bs].sort(key=lambda x: x["host_hit"])

        # L3 model is available only in profile V3.
        self.l3_restore_curve = []
        self.l3_recompute_curve = []
        self.l3_io_backlog_ms_per_token = 0.0

        if self.version == 3:
            l3 = self.profile.get("l3")
            if not l3:
                raise ValueError("HiCache cost profile version 3 requires 'l3' section")

            self.l3_restore_curve = sorted(
                l3["restore_base"],
                key=lambda x: x["storage_hit"],
            )
            self.l3_recompute_curve = sorted(
                l3["recompute_base"],
                key=lambda x: x["storage_hit"],
            )
            self.l3_io_backlog_ms_per_token = float(l3["io_backlog_ms_per_token"])

    def _nearest_running_bs(self, running_bs: int) -> int:
        return min(self.running_bs_buckets, key=lambda x: abs(x - running_bs))

    @staticmethod
    def _interp(curve, host_hit):
        if not curve:
            return None

        min_hit = curve[0]["host_hit"]
        max_hit = curve[-1]["host_hit"]

        if host_hit < min_hit or host_hit > max_hit:
            return None

        for point in curve:
            if point["host_hit"] == host_hit:
                return point["latency_ms"]

        for left, right in zip(curve, curve[1:]):
            if left["host_hit"] < host_hit < right["host_hit"]:
                x0, x1 = left["host_hit"], right["host_hit"]
                y0, y1 = left["latency_ms"], right["latency_ms"]
                ratio = (host_hit - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)

        return None

    @staticmethod
    def _interp_l3(curve, storage_hit):
        if not curve:
            return None

        min_hit = curve[0]["storage_hit"]
        max_hit = curve[-1]["storage_hit"]

        if storage_hit < min_hit or storage_hit > max_hit:
            return None

        for point in curve:
            if point["storage_hit"] == storage_hit:
                return point["latency_ms"]

        for left, right in zip(curve, curve[1:]):
            if left["storage_hit"] < storage_hit < right["storage_hit"]:
                x0, x1 = left["storage_hit"], right["storage_hit"]
                y0, y1 = left["latency_ms"], right["latency_ms"]
                ratio = (storage_hit - x0) / (x1 - x0)
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

        if restore_ms is None or recompute_ms is None:
            return True, restore_ms, recompute_ms

        should_recompute = recompute_ms + self.safety_margin_ms < restore_ms
        return not should_recompute, restore_ms, recompute_ms

    def estimate_l3(
        self,
        storage_hit: int,
        io_pending_tokens: int,
        query_ms: float,
    ):
        if self.version < 3:
            return None, None

        restore_base_ms = self._interp_l3(
            self.l3_restore_curve,
            storage_hit,
        )
        recompute_base_ms = self._interp_l3(
            self.l3_recompute_curve,
            storage_hit,
        )

        if restore_base_ms is None or recompute_base_ms is None:
            return None, None

        backlog_ms = max(0, io_pending_tokens) * self.l3_io_backlog_ms_per_token

        # Restore baseline already contains one L3 metadata query.
        restore_ms = restore_base_ms + backlog_ms

        # True-recompute baseline skips L3 completely, while an early cost-model
        # decision must first pay the metadata query used to discover L3 hits.
        recompute_ms = recompute_base_ms + max(0.0, query_ms)

        return restore_ms, recompute_ms

    def decide_l3(
        self,
        storage_hit: int,
        io_pending_tokens: int,
        query_ms: float,
    ):
        restore_ms, recompute_ms = self.estimate_l3(
            storage_hit,
            io_pending_tokens,
            query_ms,
        )

        # OOD / V2 profile: preserve native restore behavior.
        if restore_ms is None or recompute_ms is None:
            return True, restore_ms, recompute_ms

        should_recompute = recompute_ms + self.safety_margin_ms < restore_ms

        return not should_recompute, restore_ms, recompute_ms
