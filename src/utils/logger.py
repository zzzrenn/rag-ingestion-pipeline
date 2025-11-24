import time
import functools
import asyncio
from loguru import logger
import sys

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

def time_execution(func):
    """
    Decorator to track execution time of functions (sync and async).
    Logs the duration in seconds.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"'{func.__module__}.{func.__qualname__}' executed in {duration:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"'{func.__module__}.{func.__qualname__}' failed after {duration:.4f} seconds with error: {e}")
            raise e

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"'{func.__module__}.{func.__qualname__}' executed in {duration:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"'{func.__module__}.{func.__qualname__}' failed after {duration:.4f} seconds with error: {e}")
            raise e

    if asyncio.iscoroutinefunction(func):
        return wrapper
    else:
        return sync_wrapper
