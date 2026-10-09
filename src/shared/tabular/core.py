"""Core tabular utilities with no heavy data-analysis dependencies."""

import multiprocessing
import time
from datetime import UTC, datetime
from functools import wraps

from IPython import get_ipython

from shared.assertions import expect_true


def nearest_odd(number):
    # Ensure the returned value is odd and at least 3
    """Find nearest odd."""
    odd = number if number % 2 == 1 else number + 1
    return max(odd, 3)


def calculate_time_duration(begin_time, end_time):
    """Calculate time duration."""
    tot_secs = end_time - begin_time
    hours, remainder = divmod(tot_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    return duration


def time_decorator(func):
    """Time decorator."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.time()
        begin_datetime = datetime.fromtimestamp(begin_time, UTC).astimezone()
        text2 = f"Function <{func.__name__}> started on {begin_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        text1 = "=" * (len(text2) + 6)
        text = text1 + "\n" + text2
        print_formatted_txt(text, "RUN")

        output = func(*args, **kwargs)

        end_time = time.time()
        end_datetime = datetime.fromtimestamp(end_time, UTC).astimezone()
        duration = calculate_time_duration(begin_time, end_time)
        text1 = (
            f"Execution time of function <{func.__name__}> ended on {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} - "
            f"duration: {duration}"
        )
        text2 = "=" * (len(text1) + 6)
        text = text1 + "\n" + text2
        print_formatted_txt(text, "RUN")

        return output

    return wrapper


def reset_index(*dfs):
    """Reset index."""
    for df in dfs:
        df.reset_index(inplace=True, drop=True)


def print_formatted_txt(msg, txt_format="SECTION"):
    """Print formatted txt."""
    print()
    if msg.lower().startswith("start"):
        print(100 * "-")
    format_dict = {
        "RUN": "\033[94m",  # blue text for runs
        "SECTION": "\033[1;30;46m",  # text for sections
        "SUBSECTION": "\033[1;30m",  # subsections
        "RESULTS": "\033[92m",  # green text for results
        "WARNING": "\033[0;31m",  # critical text as warning
        "ERROR": "\033[0;31;40m",  # critical text as error - the code stops
        "END": "\033[0m",
    }
    print(f"{format_dict[txt_format]}{msg}{format_dict['END']}")
    if msg.lower().startswith("end"):
        print(100 * "*")
        print()


def check_needed_df_columns(df, needed_columns):
    """Check needed df columns."""
    not_found_columns = []
    for column in needed_columns:
        if column not in df.columns:
            not_found_columns.append(column)
    if len(not_found_columns) > 0:
        raise KeyError(f"Error: {not_found_columns} not in dataframe.")


def move_col_to_last(df, col_name, inplace=False):
    """Move col to last."""
    if not inplace:
        df = df.copy()
    expect_true(col_name in df.columns, f"Column: {col_name} does not exist in dataframe with columns: {df.columns!s}")
    if df.columns[-1] == col_name:
        pass
    else:
        col = df.pop(col_name)
        df.insert(len(df.columns), col_name, col)
    if not inplace:
        return df
    return None


def fill_empty_string(val, num_spaces=10):
    """Fill empty string."""
    return str(val).ljust(num_spaces)


def highlight_df_rows(index_names, color="yellow"):
    """Highlight df rows."""

    def highlight(row):
        if row.name in index_names:
            return [f"font-weight: bold; background-color: {color}"] * len(row)
        return [""] * len(row)

    return highlight


def get_notebook_name():
    """Get notebook name."""
    if multiprocessing.current_process().name != "MainProcess":
        return None  # Return a default value or handle it differently in the multiprocessing context

    ip = get_ipython()
    if ip is None:
        return None  # Return a default value or handle it differently when not in IPython

    if "__vsc_ipynb_file__" in ip.user_ns:
        return ip.user_ns["__vsc_ipynb_file__"]
    return None


def remove_special_characters_from_text(text, keep_chars=".,"):
    # Create a translation table
    """Remove special characters from text."""
    chars_to_remove = "".join(c for c in set(text) if not c.isalnum() and c not in keep_chars)
    trans_table = str.maketrans("", "", chars_to_remove)

    # Apply the translation
    return text.translate(trans_table)
