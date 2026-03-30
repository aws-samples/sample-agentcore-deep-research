# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Strands @tool wrapper for AgentCore Code Interpreter."""

import os

from strands import tool

from tools.code_interpreter.code_interpreter_tools import CodeInterpreterTools

_interpreter = None


def _get_interpreter() -> CodeInterpreterTools:
    global _interpreter
    if _interpreter is None:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        _interpreter = CodeInterpreterTools(region)
    return _interpreter


@tool
def execute_python(code: str) -> str:
    """Execute Python code for data visualization and statistical analysis ONLY.

    Use this tool exclusively when the user clicks "Analyze" to generate charts
    and run statistical tests based on the Data Analysis Recommendations section.
    Do NOT use this tool during the research or report writing phases.

    The sandbox has numpy, pandas, matplotlib, seaborn, scipy, and scikit-learn.
    Save charts to /tmp/ as PNG: plt.savefig('/tmp/chart_name.png', dpi=150, bbox_inches='tight')
    Always call plt.close() after each chart.

    Args:
        code: Python code for data analysis or visualization. Include all imports.
    """
    return _get_interpreter().execute_python_securely(code)
