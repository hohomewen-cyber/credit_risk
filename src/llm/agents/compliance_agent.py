"""政策合规检查 Agent。"""
import json
import re


class ComplianceAgent:
    DEFAULT_RULES = [
        "贷款利率不得超过 LPR 的 4 倍",
        "对 60 岁以上申请人，需要额外的还款能力证明",
        "贷款用途禁止用于房地产投资",
    ]

    def __init__(self, client, prompt_manager, config):
        self.client = client
        self.pm = prompt_manager
        self.config = config

    def check(self, features, decision, interest_rate, rules=None):
        prompt = self.pm.render(
            "compliance_check",
            features=features,
            decision=decision,
            interest_rate=interest_rate,
            rules=rules or self.DEFAULT_RULES,
        )
        last_error = None
        for _ in range(2):
            response = self.client.chat(
                prompt=prompt,
                system_prompt="你是银行合规审查员，只输出合法 JSON，不要输出分析过程",
                temperature=self.config.get("temperature", 0.1),
                max_tokens=self.config.get("max_tokens", 2000),
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            if not response or not response.strip():
                last_error = RuntimeError("DeepSeek 返回空正文，未生成合规检查 JSON")
                continue
            try:
                return json.loads(self._extract_json(response))
            except json.JSONDecodeError as exc:
                last_error = exc

        raise RuntimeError("DeepSeek 未返回可解析的合规检查 JSON") from last_error

    @staticmethod
    def _extract_json(text):
        # 提取被 ```json ... ``` 包裹的内容
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        return match.group(1) if match else text
