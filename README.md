# 智能信贷风控分析系统

基于 XGBoost、PyTorch MLP、Random Forest、SHAP 和 DeepSeek LLM 的 Streamlit 信贷风控课程项目。

## 功能

- 原有数据探索、特征工程、模型训练、单笔预测、SHAP解释及LLM报告
- 7.1 Random Forest
- 7.2 政策合规检查 Agent
- 7.3 Lending Club Adapter
- 7.4 KS 模型漂移监控
- 7.5 一致性哈希 A/B 测试与日志分析
- 7.6 KNN 缺失值填充
- 7.7 全部扩展功能融合到原 Streamlit 页面

## 本地运行

```powershell
D:\Anaconda\envs\credit_risk\python.exe -m streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud

- 入口文件：`app/streamlit_app.py`
- Python：3.11
- 在应用 Secrets 中配置：

```toml
DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
```

真实密钥不得写入仓库。未配置 Secret 时，机器学习、SHAP、KS 和 A/B 功能仍可使用，LLM/Agent 功能会明确提示未配置。

> 本项目仅用于学习研究，输出不构成金融建议。
