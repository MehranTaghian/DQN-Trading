# BUllish candle stick patterns:


def is_hammer(candle, percentage_of_upper_shadow=0.1, upper_bound_hammer_significance_level=0.2,
              lower_bound_hammer_significance_level=0.05):
    """
    :param candle:
    :param percentage_of_upper_shadow: upper bound of the length of the upper shadow in compared to the total length
    :param lower_bound_hammer_significance_level: the lower bound of the length of the hammer compared to the total
    length so that the hammer would not be a doji
    :param upper_bound_hammer_significance_level: the upper bound of the length of the hammer compared to the total length
    """
    if is_bullish(candle):
        total_length = candle.high - candle.low
        hammer_length = candle.close - candle.open
        upper_shadow_length = candle.high - candle.close
        if (upper_shadow_length <= percentage_of_upper_shadow * total_length) and \
                (lower_bound_hammer_significance_level * total_length <= hammer_length <=
                 upper_bound_hammer_significance_level * total_length):
            return True
    return False


def is_inverse_hammer(candle, percentage_of_lower_shadow=0.1,
                      upper_bound_hammer_significance_level=0.2, lower_bound_hammer_significance_level=0.05):
    """
    :param candle:
    :param percentage_of_lower_shadow: upper bound of the length of the upper shadow in compared to the total length
    :param lower_bound_hammer_significance_level: the lower bound of the length of the hammer compared to the total
    length so that the hammer would not be a doji
    :param upper_bound_hammer_significance_level: the upper bound of the length of the hammer compared to the total length
    """
    if is_bullish(candle):
        total_length = candle.high - candle.low
        hammer_length = candle.close - candle.open
        lower_shadow_length = candle.open - candle.low
        if (lower_shadow_length <= percentage_of_lower_shadow * total_length) and \
                (lower_bound_hammer_significance_level * total_length <= hammer_length
                 <= upper_bound_hammer_significance_level * total_length):
            return True
    return False


def is_bullish_engulfing(candle1, candle2, average_length_of_candles_bodies, significance_level=0.4):
    if (candle2.open <= candle1.close <= candle2.close) and \
            (candle2.open <= candle1.open <= candle2.close) and \
            (candle1.close > candle2.open or candle1.open < candle2.close) and \
            is_bearish(candle1) and is_significant(candle2, average_length_of_candles_bodies, significance_level):
        return True
    return False


def is_piercing_line(candle1, candle2, average_length_of_candles_bodies, significance_level=0.4,
                     gap_significance_level=0.05):
    """
    :param candle1:
    :param candle2:
    :param average_length_of_candles_bodies:
    :param significance_level:
    :param gap_significance_level: the level of significance of the gap between the closing price of the first
    candle and the opening price of the second one
    :return:
    """
    if is_bullish(candle2) and is_bearish(candle1):
        mid_point = (candle1.open + candle1.close) / 2
        candle1_length = candle1.open - candle1.close
        candle2_length = candle2.close - candle2.open

        if (gap_significance_level * max(candle1_length, candle2_length) <= (candle1.close - candle2.open)) \
                and (candle2.close >= mid_point) and is_significant(candle1, average_length_of_candles_bodies,
                                                                    significance_level) \
                and is_significant(candle2, average_length_of_candles_bodies, significance_level):
            return True
    return False


def is_morning_star(candle1, candle2, candle3, average_length_of_candles_bodies, doji_length_percentage=0.1,
                    significance_level=0.4,
                    down_percentage=0.1):
    """
    :param candle1: red large candle
    :param candle2: doji star candle
    :param candle3: green large candle
    :param average_length_of_candles_bodies:
    :param down_percentage: The percentage of the down part of two bigger candle sticks that the doji candle stick can exist
    :param doji_length_percentage: the length of the candle to be considered as doji
    :param significance_level: the length of the candle1 and 2 in order for them to be significantly large candles
    :return:
    """
    if is_significant(candle1, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle3, average_length_of_candles_bodies, significance_level) \
            and is_doji(candle2, doji_length_percentage) \
            and is_bearish(candle1) \
            and is_bullish(candle3):
        min_doji = min(candle2.open, candle2.close)
        if min_doji < candle1.close + down_percentage * (candle1.open - candle1.close) \
                and min_doji < candle3.open + down_percentage * (candle3.close - candle3.open):
            return True

    return False


def is_three_white_soldier(candle1, candle2, candle3, average_length_of_candles_bodies, significance_level=0.4):
    return is_significant(candle1, average_length_of_candles_bodies, significance_level) \
           and is_significant(candle2, average_length_of_candles_bodies, significance_level) \
           and is_significant(candle3, average_length_of_candles_bodies, significance_level) \
           and is_bullish(candle1) and is_bullish(candle2) and is_bullish(candle3)


def is_bullish_harami(candle1, candle2, max_length_candle, significance_level=0.4, gap_significance_level=0.1):
    if is_bearish(candle1) and is_bullish(candle2) \
            and is_significant(candle1, max_length_candle, significance_level):
        gap_up_amount = candle2.open - candle1.close
        candle1_length = candle1.open - candle1.close
        if gap_up_amount >= gap_significance_level * candle1_length and candle2.close <= candle1.open:
            return True
    return False


# Bearish candle stick patterns:
def is_hanging_man(candle, percentage_of_upper_shadow=0.1, lower_bound_hangman_significance_level=0.05,
                   upper_bound_hangman_significance_level=0.2):
    """
    :param candle:
    :param percentage_of_upper_shadow: upper bound of the length of the upper shadow in compared to the total length
    :param lower_bound_hangman_significance_level: the lower bound of the length of the hammer compared to the total
    length so that the hammer would not be a doji
    :param upper_bound_hangman_significance_level: the upper bound of the length of the hammer compared to the total length
    :return:
    """
    if is_bearish(candle):
        total_length = candle.high - candle.low
        hangman_length = candle.open - candle.close
        upper_shadow_length = candle.high - candle.open
        if (upper_shadow_length <= percentage_of_upper_shadow * total_length) and \
                (lower_bound_hangman_significance_level * total_length <= hangman_length
                 <= upper_bound_hangman_significance_level * total_length):
            return True
    return False


def is_shooting_star(candle, percentage_of_lower_shadow=0.1, upper_bound_hangman_significance_level=0.2,
                     lower_bound_hangman_significance_level=0.05):
    """
        :param candle:
        :param percentage_of_lower_shadow: upper bound of the length of the lower shadow in compared to the total length
        :param lower_bound_hangman_significance_level: the lower bound of the length of the hammer compared to the total
        length so that the hammer would not be a doji
        :param upper_bound_hangman_significance_level: the upper bound of the length of the hammer compared to the total length
        :return:
        """
    if is_bearish(candle):
        total_length = candle.high - candle.low
        hangman_length = candle.open - candle.close
        lower_shadow_length = candle.close - candle.low
        if (lower_shadow_length <= percentage_of_lower_shadow * total_length) and \
                (lower_bound_hangman_significance_level * total_length <= hangman_length <=
                 upper_bound_hangman_significance_level * total_length):
            return True
    return False


def is_bearish_engulfing(candle1, candle2, average_length_of_candles_bodies, significance_level=0.4):
    if (candle2.close <= candle1.close <= candle2.open) and \
            (candle2.close <= candle1.open <= candle2.open) and \
            (candle1.close < candle2.open or candle1.open > candle2.close) and \
            is_bullish(candle1) and is_significant(candle2, average_length_of_candles_bodies, significance_level):
        return True
    return False


def is_evening_star(candle1, candle2, candle3, average_length_of_candles_bodies, doji_length_percentage=0.1,
                    significance_level=0.4,
                    up_percentage=0.1):
    """
    :param candle1: red large candle
    :param candle2: doji star candle
    :param candle3: green large candle
    :param average_length_of_candles_bodies:
    :param up_percentage: The percentage of the upper part of two bigger candle stick that the doji candle stick can exist
    :param doji_length_percentage: the length of the candle to be considered as doji
    :param significance_level: the difference
    between the value of open and close of in candle1 and candle2 in order for them to be significantly large candles
    :return:
    """
    if is_significant(candle1, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle3, average_length_of_candles_bodies, significance_level) \
            and is_doji(candle2, doji_length_percentage) \
            and is_bearish(candle3) \
            and is_bullish(candle1):
        upper_part_of_doji = max(candle2.open, candle2.close)
        if (upper_part_of_doji > candle1.close - up_percentage * (candle1.close - candle1.open)) \
                and (upper_part_of_doji > candle3.open - up_percentage * (candle3.open - candle3.close)):
            return True

    return False


def is_three_black_crows(candle1, candle2, candle3, average_length_of_candles_bodies, significance_level=0.4):
    return (is_significant(candle1, average_length_of_candles_bodies, significance_level)
            and is_significant(candle2, average_length_of_candles_bodies, significance_level)
            and is_significant(candle3, average_length_of_candles_bodies, significance_level)
            and is_bearish(candle3) and is_bearish(candle2) and is_bearish(candle1))


def is_dark_cloud_cover(candle1, candle2, average_length_of_candles_bodies, significance_level=0.4,
                        gap_significance_level=0.1):
    if is_bearish(candle2) and is_bullish(candle1):
        mid_point = (candle1.close + candle1.open) / 2
        candle1_length = candle1.close - candle1.open
        candle2_length = candle2.open - candle2.close

        if (gap_significance_level * max(candle1_length, candle2_length) <= (candle2.open - candle1.close)) \
                and (candle2.close <= mid_point) and is_significant(candle1, average_length_of_candles_bodies,
                                                                    significance_level) \
                and is_significant(candle2, average_length_of_candles_bodies, significance_level):
            return True
    return False


def is_bearish_harami(candle1, candle2, max_length_candle, significance_level=0.4, gap_significance_level=0.1):
    if is_bullish(candle1) and is_bearish(candle2) \
            and is_significant(candle1, max_length_candle, significance_level):
        gap_down_amount = candle1.close - candle2.open
        candle1_length = candle1.close - candle1.open
        if gap_down_amount >= gap_significance_level * candle1_length and candle2.close >= candle1.open:
            return True
    return False


# Continuation candle stick Patterns
def is_doji(candle, average_length_of_candles_bodies, percentage_length=0.1):
    return abs(candle.open - candle.close) <= percentage_length * average_length_of_candles_bodies


def is_spinning_top(candle, average_length_of_candles_bodies, significance_level=0.2, doji_level=0.3, offset=0.05):
    """
    :param candle:
    :param average_length_of_candles_bodies:
    :param significance_level: min length of the body
    :param doji_level: max length of the body
    :param offset: the offset of the difference of the two upper and lower shadows
    :return:
    """
    if is_bullish(candle):
        upper_shadow_length = candle.high - candle.close
        lower_shadow_length = candle.open - candle.low
    else:
        upper_shadow_length = candle.high - candle.open
        lower_shadow_length = candle.close - candle.low

        # if upper-shadow and lower shadow are almost the same length
    return is_significant(candle, average_length_of_candles_bodies, significance_level) and is_doji(candle,
                                                                                                    doji_level) and \
           abs(upper_shadow_length - lower_shadow_length) <= offset


def is_falling_three_methods(candle1, candle2, candle3, candle4, candle5, average_length_of_candles_bodies,
                             significance_level=0.4):
    if is_bearish(candle1) and is_bearish(candle5) and \
            is_bullish(candle2) and is_bullish(candle3) \
            and is_bullish(candle4) \
            and is_significant(candle1, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle2, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle3, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle4, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle5, average_length_of_candles_bodies, significance_level):
        max_three_middle_candles = max(candle2.close, candle3.close, candle4.close)
        min_three_middle_candles = min(candle2.open, candle3.open, candle4.open)
        return max_three_middle_candles <= candle1.high and min_three_middle_candles >= candle5.low

    return False


def is_rising_three_methods(candle1, candle2, candle3, candle4, candle5, average_length_of_candles_bodies,
                            significance_level=0.4):
    if is_bullish(candle1) and is_bullish(candle5) and \
            is_bearish(candle2) and is_bearish(candle3) \
            and is_bearish(candle4) and is_significant(candle1, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle2, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle3, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle4, average_length_of_candles_bodies, significance_level) \
            and is_significant(candle5, average_length_of_candles_bodies, significance_level):
        max_three_middle_candles = max(candle2.open, candle3.open, candle4.open)
        min_three_middle_candles = min(candle2.close, candle3.close, candle4.close)
        return max_three_middle_candles <= candle5.high and min_three_middle_candles >= candle1.low
    return False


def is_bearish(candle):
    return candle.close <= candle.open


def is_bullish(candle):
    return candle.close >= candle.open


def is_significant(candle, average_length_of_candles_bodies, significant_level):
    return abs(candle.open - candle.close) >= significant_level * average_length_of_candles_bodies
