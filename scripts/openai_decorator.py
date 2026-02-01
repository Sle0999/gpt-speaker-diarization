import functools
import time

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


def retry_on_openai_errors(max_retry):
    """
    this function retries the function that it decorates in case of openai errors (RateLimitError, Timeout, APIError, APIConnectionError, InternalServerError, BadRequestError)`
    :param max_retry: the maximum number of retries
    :return: the decorator
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry = 0
            while retry < max_retry:
                try:
                    return func(*args, **kwargs)
                except (
                    APITimeoutError,
                    APIError,
                    APIConnectionError,
                    InternalServerError,
                    BadRequestError,
                    RateLimitError,
                ) as error:
                    retry += 1
                    time.sleep(0.006)
                    print(f"Retrying {retry} time due to error: {error}")
            raise Exception(f"Reached maximum number of retries ({max_retry})")
        return wrapper
    return decorator
