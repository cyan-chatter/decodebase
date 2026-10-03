import pytest

from cfl.core.retry import with_retries


def test_success_after_transient_failures_preserves_arguments_and_metadata():
    calls = []

    @with_retries(max_retries=2)
    def operation(value: int, *, increment: int) -> int:
        """Perform an operation."""
        calls.append((value, increment))
        if len(calls) < 3:
            raise RuntimeError("temporarily unavailable")
        return value + increment

    assert operation(4, increment=2) == 6
    assert calls == [(4, 2)] * 3
    assert operation.__name__ == "operation"
    assert operation.__doc__ == "Perform an operation."
    assert operation.__wrapped__.__name__ == "operation"


@pytest.mark.parametrize("max_retries", [0, 1, 3])
def test_exhaustion_obeys_limit_and_preserves_final_exception(max_retries):
    errors = []

    @with_retries(max_retries=max_retries)
    def operation():
        error = RuntimeError(f"failure {len(errors) + 1}")
        errors.append(error)
        raise error

    with pytest.raises(RuntimeError) as raised:
        operation()
    assert len(errors) == max_retries + 1
    assert raised.value is errors[-1]


def test_successful_call_is_not_repeated_and_each_invocation_has_its_own_limit():
    calls = 0

    @with_retries(max_retries=1)
    def operation():
        nonlocal calls
        calls += 1
        if calls % 2:
            raise RuntimeError("transient")

    assert operation() is None and calls == 2
    assert operation() is None and calls == 4


def test_unmatched_error_is_not_retried():
    calls = 0

    @with_retries(max_retries=3)
    def operation():
        nonlocal calls
        calls += 1
        raise ValueError("invalid input")

    with pytest.raises(ValueError, match="invalid input"):
        operation()
    assert calls == 1


def test_configured_exception_types_are_retried():
    calls = 0

    @with_retries(max_retries=2, retry_on=(TimeoutError, ConnectionError))
    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError()
        if calls == 2:
            raise ConnectionError()
        return "ready"

    assert operation() == "ready" and calls == 3


@pytest.mark.parametrize("value", [-1, 1.5, "3", True, None])
def test_invalid_retry_limit_rejected(value):
    with pytest.raises(ValueError if value == -1 else TypeError, match="max_retries"):
        with_retries(max_retries=value)


@pytest.mark.parametrize("value", [(), RuntimeError, ("RuntimeError",), (KeyboardInterrupt,)])
def test_invalid_exception_types_rejected(value):
    with pytest.raises(TypeError, match="retry_on"):
        with_retries(retry_on=value)
