import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear slowapi's in-memory counters before every test so rate-limit
    state from one test cannot bleed into the next."""
    from api.main import limiter

    limiter._storage.reset()
    yield
