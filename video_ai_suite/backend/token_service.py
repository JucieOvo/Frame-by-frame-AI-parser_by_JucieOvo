"""
模块名称：token_service
功能描述：
    提供 Token 使用量与费用计算能力。
    该模块只负责纯数据计算，不依赖 Streamlit 页面对象，便于界面层和其他入口复用。

主要组件：
    - calculate_token_cost: 根据阶梯价格计算费用。
    - create_empty_token_usage: 构造空的统计结构。
    - accumulate_token_usage: 生成更新后的统计结果。

依赖说明：
    - Python 标准库: 无额外外部依赖。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从原始单文件应用中抽离 Token 统计逻辑。
"""

from __future__ import annotations


def create_empty_token_usage() -> dict[str, float]:
    """
    创建空的 Token 统计结构。

    :return: 初始统计字典。
    """
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
    }


def calculate_token_cost(input_tokens: int, output_tokens: int) -> dict[str, float]:
    """
    根据阶梯价格计算 Token 费用。

    :param input_tokens: 输入 Token 数量。
    :param output_tokens: 输出 Token 数量。
    :return: 输入费用、输出费用与总费用。
    """

    def calculate_tiered_cost(
        tokens: int,
        price_tiers: list[tuple[tuple[int, int | None], float]],
    ) -> float:
        """
        按阶梯价格累加 Token 费用。

        :param tokens: 待计算的 Token 数量。
        :param price_tiers: 阶梯区间与单价配置。
        :return: 当前 Token 数量对应的累计费用。
        """
        cost = 0.0
        remaining = tokens

        for (tier_start, tier_end), price_per_k in price_tiers:
            if remaining <= 0:
                break

            tier_size = tier_end - tier_start if tier_end is not None else remaining
            tokens_in_tier = min(remaining, tier_size)
            cost += (tokens_in_tier / 1000) * price_per_k
            remaining -= tokens_in_tier

        return cost

    input_tiers = [
        ((0, 32000), 0.001),
        ((32000, 128000), 0.0015),
        ((128000, 256000), 0.003),
    ]
    output_tiers = [
        ((0, 32000), 0.01),
        ((32000, 128000), 0.015),
        ((128000, 256000), 0.03),
    ]

    input_cost = calculate_tiered_cost(input_tokens, input_tiers)
    output_cost = calculate_tiered_cost(output_tokens, output_tiers)

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def accumulate_token_usage(
    current_usage: dict[str, float],
    input_tokens: int,
    output_tokens: int,
) -> dict[str, float]:
    """
    返回更新后的 Token 统计结构。

    :param current_usage: 当前统计结构。
    :param input_tokens: 本次新增输入 Token 数量。
    :param output_tokens: 本次新增输出 Token 数量。
    :return: 更新后的统计结构。
    """
    updated_usage = {
        "input_tokens": int(current_usage.get("input_tokens", 0)) + input_tokens,
        "output_tokens": int(current_usage.get("output_tokens", 0)) + output_tokens,
        "total_cost": 0.0,
    }

    cost_info = calculate_token_cost(
        updated_usage["input_tokens"],
        updated_usage["output_tokens"],
    )
    updated_usage["total_cost"] = cost_info["total_cost"]
    return updated_usage
