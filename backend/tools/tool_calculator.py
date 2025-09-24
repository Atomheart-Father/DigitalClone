"""
Calculator tool for the Digital Clone AI Assistant.

Provides safe mathematical expression evaluation using AST parsing
to avoid security risks associated with eval().
"""

import ast
import operator
import math
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "calculator",
    "description": "安全计算器，支持基本的数学运算（加减乘除、幂运算、括号），避免使用eval的安全风险",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4' 或 '(12 + 7) * 3**2'"
            }
        },
        "required": ["expression"],
        "additionalProperties": False
    }
}


class SafeEvaluator:
    """Safe mathematical expression evaluator using AST."""

    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.pow,  # Support ^ as power operator
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Supported functions
    FUNCTIONS = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
    }

    # Supported constants
    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
    }

    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Numerical result

        Raises:
            ValueError: If expression is invalid or contains unsupported operations
        """
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode='eval')

            # Evaluate the AST safely
            return self._eval_node(tree.body)

        except SyntaxError:
            raise ValueError(f"无效的数学表达式语法: {expression}")
        except ZeroDivisionError:
            raise ValueError("除数不能为零")
        except OverflowError:
            raise ValueError("计算结果超出范围")
        except Exception as e:
            raise ValueError(f"计算错误: {str(e)}")

    def _eval_node(self, node):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):
            # Numbers and constants
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError(f"不支持的常量类型: {type(node.value)}")

        elif isinstance(node, ast.BinOp):
            # Binary operations (e.g., 2 + 3)
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的二元运算符: {type(node.op)}")
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            # Unary operations (e.g., -5)
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op)}")
            return op(operand)

        elif isinstance(node, ast.Name):
            # Constants and functions
            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]
            else:
                raise ValueError(f"不支持的名称: {node.id}")

        elif isinstance(node, ast.Call):
            # Function calls
            if not isinstance(node.func, ast.Name):
                raise ValueError("只支持简单函数调用")

            func_name = node.func.id
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"不支持的函数: {func_name}")

            # Evaluate arguments
            args = [self._eval_node(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("不支持关键字参数")

            func = self.FUNCTIONS[func_name]
            return func(*args)

        else:
            raise ValueError(f"不支持的表达式类型: {type(node)}")


_evaluator = SafeEvaluator()


def run(expression: str) -> Dict[str, Any]:
    """
    Execute calculator tool.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not expression or not expression.strip():
            return {"ok": False, "error": "表达式不能为空"}

        # Clean the expression
        expression = expression.strip()

        logger.info(f"Evaluating expression: {expression}")

        # Evaluate the expression
        result = _evaluator.evaluate(expression)

        # Format the result nicely
        if isinstance(result, float):
            # Check if it's a whole number
            if result.is_integer():
                result = int(result)
            else:
                # Round to reasonable precision
                result = round(result, 10)

        logger.info(f"Expression result: {expression} = {result}")
        return {"ok": True, "value": result}

    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"Calculator error for '{expression}': {error_msg}")
        return {"ok": False, "error": error_msg}
    except Exception as e:
        error_msg = f"意外错误: {str(e)}"
        logger.error(f"Unexpected calculator error for '{expression}': {e}")
        return {"ok": False, "error": error_msg}
