"""在独立临时仓库插入合成敏感数据，验证工作树、忽略文件及悬空Git对象都不能漏检。"""
import ipaddress
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import unittest

from support import ROOT


class PublicAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='public-audit-tests-')
        self.base = Path(self.temporary.name)
        self.root = self.base / 'source'
        shutil.copytree(ROOT / 'tools', self.root / 'tools')

    def tearDown(self):
        self.temporary.cleanup()

    def scan(self, *options):
        return subprocess.run([sys.executable, '-B', str(self.root / 'tools/audit-public.py'), *map(str, options)],
                              text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_clean_synthetic_tree_passes_and_host_literal_fails(self):
        self.assertEqual(self.scan().returncode, 0)
        # 使用文档保留网段的合成地址；源码中不保存任何真实服务器地址。
        address = str(ipaddress.IPv4Address(0xc0000201 + secrets.randbelow(200)))
        (self.root / 'unreviewed.md').write_text('endpoint=' + address, encoding='utf-8')
        result = self.scan()
        self.assertEqual(result.returncode, 1)
        self.assertIn('literal-host-address', result.stdout)
        self.assertNotIn(address, result.stdout)

    def test_ignored_environment_file_still_fails(self):
        (self.root / '.gitignore').write_text('.env\n', encoding='utf-8')
        (self.root / '.env').write_text('SYNTHETIC=true\n', encoding='utf-8')
        self.assertNotEqual(self.scan().returncode, 0)

    def test_empty_file_with_private_filename_still_fails(self):
        address = str(ipaddress.IPv4Address(0xc0000201 + secrets.randbelow(200)))
        (self.root / (address + '.txt')).write_text('', encoding='utf-8')
        result = self.scan()
        self.assertEqual(result.returncode, 1)
        self.assertIn('path-sha256:', result.stdout)
        self.assertNotIn(address, result.stdout)

    def test_json_runtime_identity_is_not_a_template(self):
        (self.root / 'snapshot.json').write_text(json.dumps({'model_id': 47}), encoding='utf-8')
        result = self.scan()
        self.assertEqual(result.returncode, 1)
        self.assertIn('runtime-json-identity', result.stdout)

    def test_unreachable_git_blob_cannot_hide_private_markers(self):
        subprocess.run(['git', 'init', '--quiet', '--template=', str(self.root)], check=True)
        self.assertEqual(self.scan('--with-git').returncode, 0)
        marker = 'synthetic-private-' + secrets.token_hex(20)
        denied = self.base / 'private-markers.json'
        denied.write_text(json.dumps([marker]), encoding='utf-8')
        subprocess.run(['git', '-C', str(self.root), 'hash-object', '-w', '--stdin'], input=marker,
                       text=True, check=True, stdout=subprocess.PIPE)
        result = self.scan('--with-git', '--deny-file', denied)
        self.assertEqual(result.returncode, 1)
        self.assertIn('git-object:', result.stdout)
        self.assertIn('private-marker', result.stdout)
        self.assertNotIn(marker, result.stdout)

    def test_symlink_is_rejected(self):
        destination = self.base / 'private.txt'
        destination.write_text('synthetic', encoding='utf-8')
        (self.root / 'leak.txt').symlink_to(destination)
        self.assertNotEqual(self.scan().returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
