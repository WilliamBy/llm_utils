# llm-study

![Dynamic TOML Badge](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FWilliamBy%2Fllm_utils%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=%24.project.requires-python&label=python)
![PyPI - Version](https://img.shields.io/pypi/v/llm-study)

Practical toolkit for LLM studies

---

## Installation
To install LLM Toolset, ensure you have Python 3.9 or higher installed, then run the following command:
```bash
pip install llm-study
```
To install nightly version:
```bash
pip install git+https://github.com/WilliamBy/llm_utils.git@dev
```

## Dependencies
LLM Toolset depends on the following libraries:
- transformers (version >= 4.50.0)

### Example: kvcache calculator
`kv_calc.py` provides basic key-value calculation functionalities.
```python
from llm_toolset.kv_calc import calculate_key_value

result = calculate_key_value(data)
```
or use in shell
```bash
$ python -m llm_study.kv_calc -h
```

## Links
- [Homepage](https://github.com/WilliamBy/llm_utils)
- [Issue Tracker](https://github.com/WilliamBy/llm_utils/issues)
