def test_semcache_error_is_exception():
    from client import SemCacheError

    assert issubclass(SemCacheError, Exception)
