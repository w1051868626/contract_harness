"""快速切片测试：内嵌测试文本，无需下载数据集

用法:
    conda activate contract-harness
    python -m debug.test_chunking.test_chunk_quick
"""
from harness.rag.knowledge_base import KnowledgeBase

CHUNK_SIZE = 150
OVERLAP = 30

samples = {
    "法律文本（逐条解析 → _chunk_law_text）": """中华人民共和国治安管理处罚法
（2005年8月28日第十届全国人民代表大会常务委员会第十七次会议通过）
第一章 总则
第一条 为维护社会治安秩序，保障公共安全，保护公民、法人和其他组织的合法权益，规范和保障公安机关及其人民警察依法履行治安管理职责，制定本法。
第二条 扰乱公共秩序，妨害公共安全，侵犯人身权利、财产权利，妨害社会管理，具有社会危害性，依照《中华人民共和国刑法》的规定构成犯罪的，依法追究刑事责任；尚不够刑事处罚的，由公安机关依照本法给予治安管理处罚。
第二章 处罚的种类和适用
第十条 治安管理处罚的种类分为：
（一）警告；
（二）罚款；
（三）行政拘留；
（四）吊销公安机关发放的许可证。
对违反治安管理的外国人，可以附加适用限期出境或者驱逐出境。
第十一条 办理治安案件所查获的毒品、淫秽物品等违禁品，赌具、赌资，吸食、注射毒品的用具以及直接用于实施违反治安管理行为的本人所有的工具，应当收缴，按照规定处理。
违反治安管理所得的财物，追缴退还被侵害人；没有被侵害人的，登记造册，公开拍卖或者按照国家有关规定处理，所得款项上缴国库。""",

    "合同文本（无第X条 → _chunk_text）": """甲方：北京科技有限公司
乙方：上海软件有限公司
双方经友好协商，就软件开发事宜达成如下协议：
第一条 甲方委托乙方开发一套客户管理系统。
第二条 开发期限为合同签订后90个工作日。
第三条 项目总金额为人民币伍拾万元整。
第四条 付款方式为合同签订后支付30%，验收后支付70%。
第五条 乙方应在交付后提供12个月免费技术支持服务。""",

    "Markdown 合同（→ _chunk_markdown）": """# 保密协议

## 第一条 保密内容
双方在合作过程中知悉的对方商业秘密。

## 第二条 保密期限
本合同终止后三年内继续有效。

## 第三条 违约责任
违反本协议的一方应赔偿对方全部损失。""",
}

def _try_chunk(text, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    # 按 _resolve_chunks 优先级尝试
    ck = KnowledgeBase._chunk_markdown(text, "test", CHUNK_SIZE, OVERLAP)
    if ck is not None:
        tag = "_chunk_markdown"
    else:
        ck = KnowledgeBase._chunk_law_text(text, "test", CHUNK_SIZE, OVERLAP)
        if ck is not None:
            tag = "_chunk_law_text"
        else:
            ck = KnowledgeBase._chunk_text(text, "test", CHUNK_SIZE, OVERLAP)
            tag = "_chunk_text"
    print(f"  分块器: {tag}")
    print(f"  切片数: {len(ck)}")
    for i, c in enumerate(ck):
        meta = {k.value if hasattr(k, 'value') else k: v for k, v in (c.metadata or {}).items()}
        print(f"\n  [{i}] ({len(c.content)} chars)")
        if meta:
            print(f"       meta: {meta}")
        print(f"       {c.content[:120].replace(chr(10),' ')}...")
    end_ok = sum(1 for c in ck if c.content.rstrip()[-1:] in "。！？；.!?；")
    print(f"\n  结尾句柄对齐: {end_ok}/{len(ck)}")


for label, text in samples.items():
    _try_chunk(text, label)
