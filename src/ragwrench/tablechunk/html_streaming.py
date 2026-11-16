"""Incremental HTML events using the buffered parser's strict normalization rules."""
import codecs
import sqlite3

from .html import _Parser, MAX_GRID_CELLS
from .html_storage import DiskStore
from .models import Table, TableChunkError


class _StreamingParser(_Parser):
    def __init__(self, storage):
        super().__init__(0, storage)
        self.prose_bytes = self.row_bytes = self.caption_bytes = 0
        self.table_buckets_start = 0

    def check(self, size):
        if size > self.storage.limit:
            self.fail('streaming HTML working block exceeds 1 MiB limit')

    def flush(self):
        super().flush()
        self.prose_bytes = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'table' and not self.depth and not self.ignored:
            self.table_buckets_start = self.storage.serial
            self.caption_bytes = 0
        if not self.ignored:
            self.row_bytes += len(self.get_starttag_text().encode('utf-8'))
            self.check(self.row_bytes)
        super().handle_starttag(tag, attrs)
        if len(self.table_tags) > 1024:
            self.fail('streaming HTML nesting exceeds 1024 tags')

    def handle_endtag(self, tag):
        outer_row = tag == 'tr' and self.depth == 1
        super().handle_endtag(tag)
        if outer_row or not self.depth:
            self.row_bytes = 0
        if len(self.rows) > MAX_GRID_CELLS:
            self.fail('expanded table exceeds 100000 cells')

    def handle_data(self, data):
        size = len(data.encode('utf-8'))
        if not self.ignored:
            if self.depth:
                self.row_bytes += size
                self.check(self.row_bytes)
                if self.in_caption:
                    self.caption_bytes += size
                    self.check(self.caption_bytes)
            else:
                self.prose_bytes += size
                self.check(self.prose_bytes)
        super().handle_data(data)

    def make_table(self):
        table = super().make_table()
        table._disk_buckets = range(self.table_buckets_start + 1, self.storage.serial + 1)
        return table


def html_blocks(source, limit):
    storage = None
    decoder = codecs.getincrementaldecoder('utf-8')()

    def drain():
        blocks, parser.blocks = parser.blocks, []
        for block in blocks:
            try:
                yield block
            finally:
                if isinstance(block, Table):
                    storage.release(block._disk_buckets)

    try:
        storage = DiskStore(limit)
        parser = _StreamingParser(storage)
        while True:
            data = source.read(65536)
            parser.feed(decoder.decode(data, final=not data))
            # HTMLParser holds incomplete tags, entities, comments and script/style.
            # _Parser feeds eagerly (EagerHTMLParser), so rawdata is the whole backlog.
            if len(parser.rawdata.encode('utf-8')) > limit:
                raise TableChunkError('streaming HTML unfinished token exceeds 1 MiB limit')
            yield from drain()
            if not data:
                break
        parser.close()
        if parser.depth:
            raise TableChunkError('unclosed HTML table')
        if parser.ignored:
            raise TableChunkError('unclosed script/style tag')
        parser.flush()
        yield from drain()
    except sqlite3.Error as exc:
        raise TableChunkError(f'HTML temporary storage: {exc}') from exc
    finally:
        if storage is not None:
            storage.close()
