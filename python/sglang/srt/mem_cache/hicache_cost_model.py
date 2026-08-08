import json

class HiCacheCostModel:
    def __init__(self, profile_path: str, safety_margin_ms: float = 1.0):
        with open(profile_path, encoding="utf-8") as f:
            self.profile = json.load(f)
        self.prefill_bucket_size = self.profile["prefill_bucket_size"]
        self.safety_margin_ms = safety_margin_ms

    def _bucket(self, x):
        s = self.prefill_bucket_size
        return ((x + s - 1) // s) * s

    def _select_curve(self, action, prefill_tokens, qload):
        target_prefill = self._bucket(prefill_tokens)
        entries = self.profile["entries"][action]
        if not entries:
            raise RuntimeError(f"No profile entries for action={action}")

        groups = {}
        for e in entries:
            groups.setdefault((e["prefill_bucket"], e["qload"]), []).append(e)

        key = min(
            groups,
            key=lambda k: abs(k[0] - target_prefill) / self.prefill_bucket_size + abs(k[1] - qload),
        )
        return sorted(groups[key], key=lambda x: x["host_hit"])

    @staticmethod
    def _interp(curve, host_hit):
        if host_hit <= curve[0]["host_hit"]:
            return curve[0]["latency_ms"]
        if host_hit >= curve[-1]["host_hit"]:
            return curve[-1]["latency_ms"]

        for left, right in zip(curve, curve[1:]):
            if left["host_hit"] <= host_hit <= right["host_hit"]:
                x0, x1 = left["host_hit"], right["host_hit"]
                y0, y1 = left["latency_ms"], right["latency_ms"]
                ratio = (host_hit - x0) / max(x1 - x0, 1)
                return y0 + ratio * (y1 - y0)

        return curve[-1]["latency_ms"]

    def estimate(self, host_hit, prefill_batch_tokens, qload):
        restore_curve = self._select_curve("restore", prefill_batch_tokens, qload)
        recompute_curve = self._select_curve("recompute", prefill_batch_tokens, qload)
        restore_ms = self._interp(restore_curve, host_hit)
        recompute_ms = self._interp(recompute_curve, host_hit)
        return restore_ms, recompute_ms

    def decide(self, host_hit, prefill_batch_tokens, qload):
        restore_ms, recompute_ms = self.estimate(host_hit, prefill_batch_tokens, qload)
        should_restore = restore_ms + self.safety_margin_ms < recompute_ms
        return should_restore, restore_ms, recompute_ms
