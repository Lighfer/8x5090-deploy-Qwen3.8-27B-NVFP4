"""验证异步发布的外部缓存计数，等待不能替代真实命中增加。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from support import generated_fixture


class CacheMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='cache-metric-tests-')
        root=Path(self.temporary.name)/'private'
        generated_fixture(root)
        spec=importlib.util.spec_from_file_location('migration_check',root/'verify-v032-migration.py')
        self.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def tearDown(self):
        self.temporary.cleanup()

    def test_delayed_counter_is_read_without_replaying_requests(self):
        with mock.patch.object(self.module,'external_hits',side_effect=[(10,'a'),(10,'b'),(24,'c')]), \
             mock.patch.object(self.module.time,'sleep') as sleep:
            self.assertEqual(self.module.settled_external_hits('synthetic',10),(24,'c'))
            self.assertEqual(sleep.call_count,2)

    def test_timeout_does_not_invent_cache_hits(self):
        with mock.patch.object(self.module,'external_hits',return_value=(10,'unchanged')), \
             mock.patch.object(self.module.time,'monotonic',side_effect=[0,20]):
            after,_=self.module.settled_external_hits('synthetic',10)
            self.assertEqual(after-10,0)

    def test_already_published_counter_returns_immediately(self):
        with mock.patch.object(self.module,'external_hits',return_value=(30,'done')), \
             mock.patch.object(self.module.time,'sleep') as sleep:
            self.assertEqual(self.module.settled_external_hits('synthetic',10),(30,'done'))
            sleep.assert_not_called()
