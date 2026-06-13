from __future__ import annotations

"""评测报告生成器（JSON / Markdown / HTML）。"""

from pathlib import Path
from typing import Any

from harness.core.config import HarnessConfig
from harness.utils.io import write_json, write_text


class EvalReporter:
    """评测报告生成器，支持 JSON / Markdown / HTML 格式。"""

    def __init__(self, output_dir: str | Path | None = None):
        config = HarnessConfig()
        self._dir = Path(output_dir) if output_dir else Path(config.report_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def report_json(self, data: dict[str, Any], name: str = "eval_report") -> Path:
        """生成 JSON 格式报告。"""
        filepath = self._dir / f"{name}.json"
        write_json(filepath, data)
        return filepath

    def report_markdown(self, data: dict[str, Any], name: str = "eval_report") -> Path:
        """生成 Markdown 格式报告。"""
        lines = [
            "# 评测报告\n",
            f"- **时间**: {data.get('timestamp', 'N/A')}",
            f"- **测试数量**: {data.get('total_items', 0)}",
            "",
            "## 聚合指标",
            "",
        ]
        for metric, value in data.get("aggregated_metrics", {}).items():
            lines.append(f"- **{metric}**: {value:.2%}")

        lines.extend(["", "## 逐项结果", ""])
        for item in data.get("per_item_results", []):
            lines.append(f"### {item['document_id']}")
            for name, val in item.get("metrics", {}).items():
                lines.append(f"- {name}: {val:.2%}")

        content = "\n".join(lines)
        filepath = self._dir / f"{name}.md"
        write_text(filepath, content)
        return filepath

    def report_html(self, data: dict[str, Any], name: str = "eval_report") -> Path:
        """生成 HTML 格式报告（依赖 jinja2，否则回退为 Markdown）。"""
        try:
            from jinja2 import Template
        except ImportError:
            return self.report_markdown(data, name)

        template = Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>评测报告</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; }
h1 { border-bottom: 2px solid #333; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; }
th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
th { background: #f5f5f5; }
.metric-value { font-weight: bold; }
.pass { color: green; }
.fail { color: red; }
</style></head>
<body>
<h1>评测报告</h1>
<p><strong>时间:</strong> {{ data.timestamp }}</p>
<p><strong>测试数量:</strong> {{ data.total_items }}</p>

<h2>聚合指标</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
{% for name, value in data.aggregated_metrics.items() %}
<tr><td>{{ name }}</td><td class="metric-value">{{ "%.2f%%" | format(value * 100) }}</td></tr>
{% endfor %}
</table>

<h2>逐项结果</h2>
{% for item in data.per_item_results %}
<h3>{{ item.document_id }}</h3>
<table>
<tr><th>指标</th><th>值</th></tr>
{% for name, value in item.metrics.items() %}
<tr><td>{{ name }}</td><td>{{ "%.2f%%" | format(value * 100) }}</td></tr>
{% endfor %}
</table>
{% endfor %}
</body></html>""")

        html = template.render(data=data)
        filepath = self._dir / f"{name}.html"
        write_text(filepath, html)
        return filepath
