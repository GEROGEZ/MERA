

import re


# ==== 正则表达式（专门从字符串中提取字段） ====
# 例如："reasoning_path": "xxx"
REG_REASONING = re.compile(r'"reasoning_path"\s*:\s*"([^"]*)"')
REG_SUGGESTED = re.compile(r'"suggested_severity"\s*:\s*"([^"]*)"')

def clean_text(text):
    """去除多余符号，包括换行、转义符、前后空格"""
    if text is None:
        return ""
    return (
        text.replace("\\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
    )

def extract_field(text):
    """从字符串 text 中匹配 pattern"""
    match = REG_SUGGESTED.search(text)
    if match:
        return clean_text(match.group(1))
    return "一般"

