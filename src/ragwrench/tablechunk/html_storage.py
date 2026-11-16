"""Temporary SQLite sequences for the shared HTML normalizer (stdlib only)."""
from collections.abc import Sequence
from dataclasses import asdict
import json
import sqlite3
import tempfile

from .models import HTMLNormalization, TableChunkError


class DiskStore:
    def __init__(self, limit):
        self.directory = tempfile.TemporaryDirectory(prefix='tablechunk-html-')
        try:
            self.db = sqlite3.connect(self.directory.name + '/tables.sqlite')
            self.db.execute('PRAGMA cache_size=-1024')
            self.db.execute('PRAGMA journal_mode=OFF')
            self.db.execute('PRAGMA temp_store=FILE')
            self.db.execute('CREATE TABLE records (bucket INTEGER, key, value TEXT, cell_id TEXT, section TEXT, row_index INTEGER, PRIMARY KEY(bucket,key))')
            self.db.execute('CREATE INDEX diagnostic_cells ON records(bucket,cell_id,section,row_index)')
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.directory.cleanup()
            raise
        self.limit = limit
        self.serial = 0

    def close(self):
        try:
            self.db.close()
        finally:
            self.directory.cleanup()

    def sequence(self, kind=''):
        self.serial += 1
        return DiskList(self, self.serial, kind)

    def mapping(self):
        self.serial += 1
        return DiskMap(self, self.serial)

    def release(self, buckets):
        for bucket in buckets:
            self.db.execute('DELETE FROM records WHERE bucket=?', (bucket,))
        self.db.commit()

    def check(self, value):
        # Also bounds rows expanded by spans, headers and parser working blocks.
        if len(json.dumps(value, ensure_ascii=False).encode('utf-8')) > self.limit:
            raise TableChunkError('streaming HTML working record exceeds 1 MiB limit')


class DiskMap:
    def __init__(self, store, bucket):
        self.store, self.bucket = store, bucket

    def __setitem__(self, key, value):
        self.store.check(value)
        self.store.db.execute('INSERT OR REPLACE INTO records(bucket,key,value) VALUES(?,?,?)',
                              (self.bucket, key, json.dumps(value, ensure_ascii=False)))

    def __getitem__(self, key):
        row = self.store.db.execute('SELECT value FROM records WHERE bucket=? AND key=?', (self.bucket, key)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class DiskList(Sequence):
    def __init__(self, store, bucket, kind='', start=0, count=0):
        self.store, self.bucket, self.kind = store, bucket, kind
        self.start, self.count = start, count

    def __len__(self):
        return self.count

    def append(self, value):
        if self.kind == 'source_rows':
            value = [[asdict(cell) for cell in value[0]], *value[1:]]
        self.store.check(value)
        diagnostic = value if self.kind == 'diagnostics' else {}
        self.store.db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',
                              (self.bucket, self.start + self.count, json.dumps(value, ensure_ascii=False),
                               diagnostic.get('cell_id'), diagnostic.get('section'), diagnostic.get('row_index')))
        self.count += 1

    def decode(self, value):
        value = json.loads(value)
        if self.kind == 'source_rows':
            from .html import Cell
            value[0] = [Cell(**cell) for cell in value[0]]
        return value

    def __iter__(self):
        cursor = self.store.db.execute('SELECT value FROM records WHERE bucket=? AND key>=? AND key<? ORDER BY key',
                                      (self.bucket, self.start, self.start + self.count))
        for (value,) in cursor:
            yield self.decode(value)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, end, step = key.indices(self.count)
            if step != 1:
                return [self[i] for i in range(start, end, step)]
            return DiskList(self.store, self.bucket, self.kind, self.start + start, max(0, end - start))
        if key < 0:
            key += self.count
        if not 0 <= key < self.count:
            raise IndexError(key)
        value = self.store.db.execute('SELECT value FROM records WHERE bucket=? AND key=?',
                                     (self.bucket, self.start + key)).fetchone()[0]
        return self.decode(value)


class DiskNormalization(HTMLNormalization):
    def for_rows(self, start, end):
        rows = list(self.row_cell_ids[start:end])
        needed = self.required_cells(rows)
        diagnostics = self.diagnostics
        db, bucket = diagnostics.store.db, diagnostics.bucket
        selected = list(db.execute(
            "SELECT key,value FROM records WHERE bucket=? AND cell_id IS NULL AND section IS NULL", (bucket,)))
        selected.extend(db.execute(
            "SELECT key,value FROM records WHERE bucket=? AND cell_id IS NULL AND section='headers'", (bucket,)))
        selected.extend(db.execute(
            "SELECT key,value FROM records WHERE bucket=? AND cell_id IS NULL AND section='rows' AND row_index>=? AND row_index<?",
            (bucket, start, end)))
        for cell in needed:
            selected.extend(db.execute('SELECT key,value FROM records WHERE bucket=? AND cell_id=?', (bucket, cell)))
        return {'schema_version': 1, 'header_cell_ids': self.header_cell_ids, 'row_cell_ids': rows,
                'source_cells': {key: self.source_cells[key] for key in sorted(needed)},
                'diagnostics': [json.loads(value) for _, value in sorted(selected)]}
