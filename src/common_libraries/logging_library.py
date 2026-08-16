"""Library containing functions for logging messages."""

import common_libraries.io_library as io_l
from global_variables import LOG_SEPARATOR_DURATION
from import_libraries import *  # NOSONAR # NOSONAR
import threading

# THREAD-SCOPED file handler that captures ONLY the current thread
class _ThreadOnlyFilter(logging.Filter):
    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self.thread_id


def attach_per_product_file_handler(base_logger: logging.Logger, log_file: str) -> logging.Handler:
    """Attach a file handler for each logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    h = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s | %(message)s"))
    h.addFilter(_ThreadOnlyFilter(threading.get_ident()))
    base_logger.addHandler(h)
    return h


def detach_handler(base_logger: logging.Logger, handler: logging.Handler) -> None:
    """Remove handler."""
    try:
        base_logger.removeHandler(handler)
    finally:
        handler.close()


# TEMPORARILY MUTE the current thread on a specific logger (to avoid double writes)
class _ThreadExcludeFilter(logging.Filter):
    """Drop log records coming from a specific thread."""

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread != self.thread_id


def suppress_current_thread_on_logger(logger: logging.Logger) -> list[tuple[logging.Handler, logging.Filter]]:
    """Add a filter to every handler on `logger` so current thread's records are ignored."""
    filt = _ThreadExcludeFilter(threading.get_ident())
    applied: list[tuple[logging.Handler, logging.Filter]] = []
    for h in logger.handlers:
        h.addFilter(filt)
        applied.append((h, filt))
    return applied


def restore_suppressed_logger(applied: list[tuple[logging.Handler, logging.Filter]]) -> None:
    """Remove previously-applied thread-exclude filters from handlers."""
    for h, f in applied:
        h.removeFilter(f)


def calculate_time_duration(begin_time: float, end_time: float):
    """Calculate the elapsed time between two timestamps and formats it as HH:MM:SS.

    Parameters
    ----------
    begin_time : float
        The start time in seconds since the process start.
    end_time : float
        The end time in seconds.

    Returns
    -------
    str
        A string representing the duration in the format 'HH:MM:SS'.

    """
    tot_secs = end_time - begin_time
    hours, remainder = divmod(tot_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    return duration


def format_process_duration(begin_time: float, end_time: float, process_name: str):
    """Log the start time, end time, and duration of a process using the provided logger.

    Parameters
    ----------
    begin_time : float
        The start time in seconds.
    end_time : float
        The end time in seconds.
    process_name : str
        The name of the process to include in the log message.

    Returns
    -------
    str
        The formatted text with info about the duration.

    """
    begin_datetime = datetime.fromtimestamp(begin_time)
    end_datetime = datetime.fromtimestamp(end_time)
    duration = calculate_time_duration(begin_time, end_time)
    safe_name = process_name.replace("\\", "/")
    text1 = f"{safe_name} Started on {begin_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    text1 += f" - Ended on {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    text1 += f" (duration: {duration})"
    text = LOG_SEPARATOR_DURATION + "\n" + text1 + "\n" + LOG_SEPARATOR_DURATION
    return text


def log_process_duration(begin_time: float, end_time: float, process_name: str, logger: logging.Logger):
    """Log the start time, end time, and duration of a process using the provided logger.

    Parameters
    ----------
    begin_time : float
        The start time in seconds.
    end_time : float
        The end time in seconds.
    process_name : str
        The name of the process to include in the log message.
    logger : logging.Logger
        A configured logger instance used to write the log.

    Returns
    -------
    None

    """
    text = format_process_duration(begin_time, end_time, process_name)
    logger.info(text)


def setup_logging(log_folder: str, file_prefix: str) -> logging.Logger:
    """Set up a logger that writes to both a file and the console, avoiding duplicate handlers.

    Parameters
    ----------
    log_folder : str
        The directory where the log file should be created.
    file_prefix : str
        The prefix used to name the log file and logger instance.

    Returns
    -------
    logging.Logger
        A configured logger instance with file and console handlers.

    Raises
    ------
    Exception
        If an error occurs during logger setup, such as an invalid folder path or file access issue.

    """
    try:
        os.makedirs(log_folder, exist_ok=True)  # Ensure log folder exists

        log_file = os.path.join(log_folder, f"{file_prefix}.log")

        # Get or create the logger
        logger = logging.getLogger(file_prefix)
        logger.setLevel(logging.INFO)

        # **Check if handlers already exist to prevent duplicate logging**
        if not logger.handlers:
            # File Handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Log Format
            log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            # Attach formatter
            file_handler.setFormatter(log_format)
            console_handler.setFormatter(log_format)

            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger
    except Exception as e:
        raise RuntimeError(f"Error setting log parameters: {str(e)}") from e


def release_logger(logger: logging.Logger) -> None:
    """Release all handlers associated with a logger to free resources and prevent duplicate logs.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance whose handlers should be released.

    Returns
    -------
    None

    """
    # Get all handlers associated with the logger
    handlers = logger.handlers[:]

    # Loop through each handler and remove it from the logger
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def log_message(logger: Optional[logging.Logger] = None, msg: str = "", type: str = "info") -> None:
    """Log a message either to the console or a provided logger, with support for info and error types.

    Parameters
    ----------
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.
    msg : str
        The message to log.
    type : str, optional
        The type of log message; either 'info' or 'error'. Defaults to 'info'.

    Returns
    -------
    None

    """
    # Validate the type of log message
    if type not in ["info", "error", "warning"]:
        raise ValueError("Invalid log type. Use 'info', 'error', or 'warning'.")
    # Log the message
    if type == "error":
        print(msg) if logger is None else logger.error(msg)
    elif type == "info":
        print(msg) if logger is None else logger.info(msg)
    elif type == "warning":
        print(msg) if logger is None else logger.warning(msg)


def append_and_delete_logs(main_log_path: str, pattern: str, section_title: str) -> None:
    """Append the contents of multiple log files to a main log file and deletes the originals.

    This function searches for log files matching the given `pattern`, appends their contents
    to the specified `main_log_path` under a clearly marked section header, and deletes each
    individual log file after it has been processed.

    Parameters
    ----------
    main_log_path : str
        Full path to the main log file where all contents will be consolidated.
    pattern : str
        Glob pattern used to find individual log files (e.g., '/logs/job_*_download_*.log').
    section_title : str
        A label inserted before each file's contents to indicate its section in the main log.

    Returns
    -------
    None

    """
    os.makedirs(os.path.dirname(main_log_path), exist_ok=True)

    # Ensure the main log file exists
    matching_logs = sorted(glob(pattern, recursive=True))
    if not matching_logs:
        msg = f"No logs found for pattern: {pattern}"
        log_message(msg=msg, type="warning")
        try:
            with open(main_log_path, "a", encoding="utf-8") as main_log:
                main_log.write(f"\n[WARN] {section_title}: {msg}\n")
        except OSError as e:
            log_message(msg=f"Could not write missing-log warning to {main_log_path}: {e}", type="warning")
        return

    # Use differenet symbols to differentiate main sections
    print_symbol = "="
    # Prints summary of the Download and Product handler logs
    if section_title.startswith("PIPELINE"):
        print_symbol = "#"
    # Prints analytically the downleaders & prpoduct handlers for each product
    elif section_title.startswith(("DOWNLOAD", "PRODUCT")):
        print_symbol = "+"

    # Create the main log file if it doesn't exist
    with open(main_log_path, "a", encoding="utf-8") as main_log:
        for log_file in matching_logs:
            main_log.write(f"\n\n{print_symbol*150}\n{section_title}: {os.path.basename(log_file)}\n{print_symbol*150}\n")
            try:
                with open(log_file, "r", encoding="utf-8") as lf:
                    main_log.write(lf.read())
            except FileNotFoundError:
                main_log.write(f"[WARN] Missing per-product log: {log_file}\n")
            main_log.write(f"\n{print_symbol*150}\n")

            # Delete after appending
            try:
                io_l.delete_file(log_file)
            except Exception:
                os.remove(log_file)


def append_error_to_main_log(
    work_folder: str,
    error_msg: str,
    job_id: Optional[str] = None,
    stage: Optional[str] = None,
    main_log_name: str = "execution_process.log",
) -> None:
    """Immediately append an error message to the main execution log in an ERROR section.

    This function ensures that critical errors are captured directly in the main log file
    (EXECUTION_LOG) even if the pipeline crashes before the standard log merge process completes.
    Errors are written with clear ERROR section markers and timestamp.

    Thread-safe and process-safe: uses append mode with explicit flush.

    Parameters
    ----------
    work_folder : str
        The working folder where the main log file is located.
    error_msg : str
        The error message to log.
    job_id : str, optional
        The job identifier (e.g., 'Job_S2MSI2A') for context.
    stage : str, optional
        The pipeline stage where the error occurred (e.g., 'DOWNLOAD', 'PROCESSING').
    main_log_name : str, optional
        Name of the main log file (default: 'execution_process.log').

    Returns
    -------
    None

    """
    try:
        main_log_path = os.path.join(work_folder, main_log_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build context prefix
        context_parts = []
        if job_id:
            context_parts.append(f"JOB={job_id}")
        if stage:
            context_parts.append(f"STAGE={stage}")
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format error section
        error_section = [
            "\n",
            "!" * 150,
            f"ERROR{context_str} | {timestamp}",
            "!" * 150,
            error_msg,
            "!" * 150,
            "\n",
        ]

        # Append to main log (create if doesn't exist)
        os.makedirs(os.path.dirname(main_log_path) if os.path.dirname(main_log_path) else ".", exist_ok=True)
        with open(main_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(error_section))
            f.flush()  # Ensure immediate write to disk
            os.fsync(f.fileno())  # Force OS-level flush

    except Exception as e:
        # Last resort: print to console if we can't write to log
        print(f"[CRITICAL] Failed to append error to main log: {e}")
        print(f"[CRITICAL] Original error: {error_msg}")


def append_progress_to_main_log(
    work_folder: str,
    progress_msg: str,
    job_id: Optional[str] = None,
    stage: Optional[str] = None,
    level: str = "INFO",
    main_log_name: str = "execution_process.log",
) -> None:
    """Immediately append a progress message to the main execution log.

    This function provides real-time visibility into pipeline progress by writing
    directly to EXECUTION_LOG at key milestones. Messages are visible even if the
    pipeline crashes before log merge completes.

    Thread-safe and process-safe: uses append mode with explicit flush.

    Parameters
    ----------
    work_folder : str
        The working folder where the main log file is located.
    progress_msg : str
        The progress message to log.
    job_id : str, optional
        The job identifier (e.g., 'Job_S2MSI2A') for context.
    stage : str, optional
        The pipeline stage (e.g., 'DOWNLOAD', 'PROCESSING').
    level : str, optional
        Log level (INFO, WARNING, etc.). Defaults to 'INFO'.
    main_log_name : str, optional
        Name of the main log file (default: 'execution_process.log').

    Returns
    -------
    None

    """
    try:
        main_log_path = os.path.join(work_folder, main_log_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build context prefix
        context_parts = [level]
        if job_id:
            context_parts.append(f"JOB={job_id}")
        if stage:
            context_parts.append(f"STAGE={stage}")
        context_str = f"[{', '.join(context_parts)}]"

        # Format progress entry (single line)
        progress_line = f"{timestamp} {context_str} {progress_msg}\n"

        # Append to main log (create if doesn't exist)
        os.makedirs(os.path.dirname(main_log_path) if os.path.dirname(main_log_path) else ".", exist_ok=True)
        with open(main_log_path, "a", encoding="utf-8") as f:
            f.write(progress_line)
            f.flush()  # Ensure immediate write to disk
            os.fsync(f.fileno())  # Force OS-level flush

    except Exception:
        print(f"[CRITICAL] Failed to append progress to main log: {progress_msg}")


def merge_logs_into_main(job_id: Optional[str], log_folder: str) -> None:
    """Merge job-specific log files into the main pipeline log file and deletes them afterward.

    This function appends the contents of all log files in the specified `log_folder`
    that start with the given `job_id` and match the following patterns:

    1. `job_id_download_*.log` — logs from the download phase.
    2. `job_id_product_handler_*.log` — logs from the product handling phase.

    Each file's contents are appended to `{job_id}_pipeline.log` with section headers.
    The original `download` and `product_handler` log files are deleted after merging.

    Parameters
    ----------
    job_id : str
        The job identifier (e.g., 'Job_20250503_093549') used as a prefix to locate related log files.
    log_folder : str
        The path to the folder where all the log files are stored.

    Returns
    -------
    None

    """
    if not job_id:
        raise ValueError("job_id cannot be None or empty.")
    if not log_folder:
        raise ValueError("log_folder cannot be None or empty.")
    if not os.path.exists(log_folder):
        io_l.create_folder(log_folder)

    # Define the main log file path
    main_log_path = os.path.join(log_folder, f"{job_id}_pipeline.log")

    # Append and delete download logs
    append_and_delete_logs(main_log_path, os.path.join(log_folder, f"{job_id}_download_*.log"), "DOWNLOAD LOG")

    # Append and delete product handler logs
    append_and_delete_logs(main_log_path, os.path.join(log_folder, f"{job_id}_product_handler_*.log"), "PRODUCT HANDLER LOG")
