"""A/B 模型路由。"""
import hashlib


class ABTestRouter:
    def __init__(self, model_a, model_b, ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.ratio = ratio

    def route(self, user_id, features):
        # 一致性哈希，同一 user 总是走同一分支
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100.0
        if bucket < self.ratio:
            return "A", self.model_a.predict_proba(features)
        return "B", self.model_b.predict_proba(features)
