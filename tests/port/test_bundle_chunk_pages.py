"""Full-bundle six-field status listing; independent document detail oracle."""
import base64
from copy import deepcopy
import hashlib
import json
import shutil

import pytest

from ragwrench.port.bundle import fingerprint, load_bundle, write_bundle
from ragwrench.port.cli import main
from ragwrench.port.common import PortError, locked
from ragwrench.port.status import import_status
from .test_import_status_bundle import interrupted, forbid_io, files_snapshot, encode_cursor, decode_cursor
from .test_skip_pages import skip_mix, CONFLICTS

STATUSES = (None, 'completed', 'pending', 'unvisited', 'skipped')


def expected_rows(path, bundle):
    full = import_status(path, bundle)
    return [row for d in full['bundle_progress']['documents']
            for row in import_status(path, bundle, document=d['source_id'], list_chunks=True)['bundle_progress']['chunks']]


def invoke(path, bundle, capsys, size=None, cursor=None, status=None):
    args = ['import-status', '--state', str(path), '--bundle', str(bundle), '--list-bundle-chunks']
    if size is not None: args += ['--bundle-chunk-page-size', str(size)]
    if cursor is not None: args += ['--bundle-chunk-cursor', cursor]
    if status is not None: args += ['--bundle-chunk-status', status]
    assert main(args) == 0
    out = capsys.readouterr()
    assert not out.err
    result = json.loads(out.out)
    assert result == import_status(path, bundle, list_bundle_chunks=True, bundle_chunk_page_size=size,
                                   bundle_chunk_cursor=cursor, bundle_chunk_status=status)
    return result


def assert_page(page, full, expected, paged=True):
    assert page['bundle_progress'].pop('chunk_details') == expected
    token = page['bundle_progress'].pop('next_bundle_chunk_cursor') if paged else None
    assert page == full
    return token


@pytest.mark.parametrize('target', [None, 'tc1'])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('status', STATUSES)
@pytest.mark.parametrize('size', [None, 1, 2, 3, 10, 10**30])
def test_bundle_chunks_contract(skip_mix, capsys, request, target, legacy, status, size):
    path, bundle, state = skip_mix
    state['documents']['a']['pending']['target'] = target
    if legacy: del state['identity']['api_profile']
    path.write_text(json.dumps(state))
    with locked(path):
        request.getfixturevalue('forbid_io')
        full = import_status(path, bundle)
        assert import_status(path, bundle, list_bundle_chunks=False, bundle_chunk_status=None,
                             bundle_chunk_page_size=None, bundle_chunk_cursor=None) == full
        rows = expected_rows(path, bundle)
        assert [(r['document_source_id'], r['source_id'], r['status']) for r in rows] == [
            ('a','c0','completed'),('a','c1','pending'),('a','c2','skipped'),('a','c3','skipped'),
            ('b','c1','skipped'),('b','c2','unvisited'),('b','c3','skipped'),
            ('c','c0','skipped'),('c','c1','skipped'),('e','c0','completed')]
        assert all(set(r) == {'document_source_id','source_id','status','target_id','pending_stage','reason'} for r in rows)
        assert rows[1]['target_id'] == target
        assert rows[1]['pending_stage'] == ('chunk_creation' if target is None else 'enabled_flag')
        rows = [r for r in rows if status is None or r['status'] == status]
        before = files_snapshot(path.parent), set(path.parent.rglob('*'))
        token, offset, seen = None, 0, []
        while True:
            page = invoke(path, bundle, capsys, size, token, status)
            actual = deepcopy(page['bundle_progress']['chunk_details'])
            assert actual == (rows if size is None else rows[offset:offset+size])
            token = assert_page(page, full, actual, size is not None)
            seen.extend(actual); offset += len(actual)
            if offset == len(rows):
                assert token is None
                break
            assert token and len(token) <= 2048
        assert seen == rows
        assert (files_snapshot(path.parent), set(path.parent.rglob('*'))) == before


@pytest.mark.parametrize('boundary', ['initial','dataset-pending','empty','all-skipped','no-skips','complete'])
@pytest.mark.parametrize('status', STATUSES)
def test_bundle_chunk_boundaries(skip_mix, capsys, request, boundary, status):
    path, bundle, state = skip_mix
    m, docs, chunks = load_bundle(bundle)
    state['documents'] = {}
    if boundary in ('initial','dataset-pending'):
        state.update(dataset=None, dataset_pending=boundary == 'dataset-pending')
    elif boundary == 'empty': docs, chunks = [], []
    elif boundary in ('all-skipped', 'no-skips', 'complete'):
        chunks = [c for c in chunks if (bool(c['metadata']) or len(c['content'].encode()) > 20) == (boundary == 'all-skipped')]
        if boundary == 'complete':
            state['documents'] = {d['id']: {'id':'t'+d['id'], 'phase':'complete', 'chunks':{
                c['id']:'t'+c['id'] for c in chunks if c['doc_id'] == d['id']}, 'pending':None} for d in docs}
    state['identity']['bundle'] = fingerprint(write_bundle(bundle,m['source'],m['settings'],docs,chunks))
    path.write_text(json.dumps(state))
    request.getfixturevalue('forbid_io')
    full = import_status(path,bundle)
    rows = [r for r in expected_rows(path,bundle) if status is None or r['status'] == status]
    assert_page(invoke(path,bundle,capsys,status=status), full, rows, False)
    assert assert_page(invoke(path,bundle,capsys,10**30,status=status), full, rows) is None


CONFLICTS = CONFLICTS + [({'list_skipped_chunks':True},['--list-skipped-chunks']),
    ({'skip_page_size':1},['--skip-page-size','1']), ({'skip_cursor':''},['--skip-cursor',''])]

@pytest.mark.parametrize('options,extra', CONFLICTS + [
    ({'bundle_chunk_page_size':0},['--bundle-chunk-page-size','0']),
    ({'bundle_chunk_page_size':-1},['--bundle-chunk-page-size','-1']),
    ({'bundle_chunk_page_size':1.5},['--bundle-chunk-page-size','1.5']),
    ({'bundle_chunk_page_size':'private'},['--bundle-chunk-page-size','private']),
    ({'bundle_chunk_cursor':''},['--bundle-chunk-cursor','']),
    *[({'bundle_chunk_status':v},['--bundle-chunk-status',v]) for v in ('','Completed',' pending','all','private')]])
def test_bundle_chunk_arguments_before_reads(tmp_path,monkeypatch,capsys,forbid_io,options,extra):
    def forbidden(*a,**k): pytest.fail('argument error read inputs')
    for name in ('read_json','read_json_snapshot','load_bundle'): monkeypatch.setattr('ragwrench.port.status.'+name,forbidden)
    with pytest.raises(PortError): import_status(tmp_path/'absent',tmp_path,list_bundle_chunks=True,**options)
    with pytest.raises(SystemExit) as exc:
        main(['import-status','--state',str(tmp_path/'absent'),'--bundle',str(tmp_path),'--list-bundle-chunks',*extra])
    assert exc.value.code == 2 and not capsys.readouterr().out


@pytest.mark.parametrize('has_bundle', [False,True])
@pytest.mark.parametrize('options,extra', [
    ({'bundle_chunk_status':'pending'},['--bundle-chunk-status','pending']),
    ({'bundle_chunk_page_size':1},['--bundle-chunk-page-size','1']),
    ({'bundle_chunk_cursor':''},['--bundle-chunk-cursor','']),
    ({'bundle_chunk_page_size':1,'bundle_chunk_cursor':'x'},['--bundle-chunk-page-size','1','--bundle-chunk-cursor','x'])])
def test_bundle_chunk_requires_listing(tmp_path,monkeypatch,capsys,forbid_io,has_bundle,options,extra):
    def forbidden(*a,**k): pytest.fail('argument error read inputs')
    for name in ('read_json','read_json_snapshot','load_bundle'): monkeypatch.setattr('ragwrench.port.status.'+name,forbidden)
    with pytest.raises(PortError): import_status(tmp_path/'absent',tmp_path if has_bundle else None,**options)
    args = ['import-status','--state',str(tmp_path/'absent'),*extra]
    if has_bundle: args += ['--bundle',str(tmp_path)]
    with pytest.raises(SystemExit) as exc: main(args)
    assert exc.value.code == 2 and not capsys.readouterr().out


def test_bundle_chunk_requires_bundle(tmp_path,capsys,forbid_io):
    with pytest.raises(PortError): import_status(tmp_path/'absent',list_bundle_chunks=True)
    with pytest.raises(SystemExit) as exc: main(['import-status','--state',str(tmp_path/'absent'),'--list-bundle-chunks'])
    assert exc.value.code == 2 and not capsys.readouterr().out


@pytest.mark.parametrize('value', [True,False,1.0,'2'])
def test_bundle_chunk_api_integer(skip_mix,forbid_io,value):
    with pytest.raises(PortError): import_status(*skip_mix[:2],list_bundle_chunks=True,bundle_chunk_page_size=value)


@pytest.mark.parametrize('case', ['empty','alphabet','unicode','oversized','truncated','padded','non-json','array',
    'noncanonical','version','bool-version','extra','kind','state','bundle','status','status-omitted','after-type',
    'after-length','after-member','after-unknown','after-cross-document','after-final','state-bytes',
    'state-order','state-unknown','state-value','chunk-v1','chunk-v2','document','skip','filtered-boundary'])
def test_bundle_chunk_cursor_rejection(skip_mix,capsys,request,case):
    path,bundle,state=skip_mix
    token=invoke(path,bundle,capsys,1)['bundle_progress']['next_bundle_chunk_cursor']
    value=decode_cursor(token)
    status=None
    if case == 'empty': token=''
    elif case == 'alphabet': token='../private'
    elif case == 'unicode': token='界'
    elif case == 'oversized': token='x'*2049
    elif case == 'truncated': token=token[:1]
    elif case == 'padded': token+='='
    elif case == 'non-json': token=base64.urlsafe_b64encode(b'private').decode().rstrip('=')
    elif case == 'array': token=encode_cursor([])
    elif case == 'noncanonical': token=base64.urlsafe_b64encode(json.dumps(value,indent=1).encode()).decode().rstrip('=')
    elif case == 'state-bytes': path.write_bytes(path.read_bytes()+b' ')
    elif case == 'state-order': path.write_text(json.dumps(state,sort_keys=True))
    elif case == 'state-unknown': path.write_text(json.dumps({**state,'extra':True}))
    elif case == 'state-value':
        state['identity']['base_url']='http://different.invalid'; path.write_text(json.dumps(state))
    elif case.startswith('chunk-'):
        token=import_status(path,bundle,document='a',list_chunks=True,page_size=1,
                            chunk_status='skipped' if case=='chunk-v2' else None)['bundle_progress']['next_cursor']
    elif case == 'skip': token=import_status(path,bundle,list_skipped_chunks=True,skip_page_size=1)['bundle_progress']['next_skip_cursor']
    elif case == 'document': token=encode_cursor({'version':1,'kind':'pending-stage-documents','scope':'bundle',
        'state':value['state'],'bundle':value['bundle'],'pending_stage':'enabled_flag','after':'a'})
    elif case == 'filtered-boundary':
        status='skipped'; value['bundle_chunk_status']=status; value['after']=['a','c0']; token=encode_cursor(value)
    elif case == 'status-omitted': del value['bundle_chunk_status']; token=encode_cursor(value)
    else:
        key,changed={'version':('version',99),'bool-version':('version',True),'extra':('extra',1),
            'kind':('kind','chunks'),'state':('state','f'*64),'bundle':('bundle','f'*64),'status':('bundle_chunk_status','skipped'),
            'after-type':('after','a/c0'),'after-length':('after',['a']),'after-member':('after',['a',2]),
            'after-unknown':('after',['unknown','c0']),'after-cross-document':('after',['b','c0']),
            'after-final':('after',['e','c0'])}[case]
        value[key]=changed; token=encode_cursor(value)
    request.getfixturevalue('forbid_io')
    before=files_snapshot(path.parent)
    with pytest.raises(PortError,match='Invalid bundle chunk continuation'):
        import_status(path,bundle,list_bundle_chunks=True,bundle_chunk_page_size=1,bundle_chunk_cursor=token,bundle_chunk_status=status)
    args=['import-status','--state',str(path),'--bundle',str(bundle),'--list-bundle-chunks',
          '--bundle-chunk-page-size','1','--bundle-chunk-cursor',token]
    if status is not None: args+=['--bundle-chunk-status',status]
    assert main(args)==1
    out=capsys.readouterr()
    assert not out.out and 'Invalid bundle chunk continuation' in out.err and 'private' not in out.err
    assert files_snapshot(path.parent)==before


def test_bundle_chunk_cursor_copies_sizes_snapshot_and_kinds(skip_mix,tmp_path,monkeypatch,capsys,forbid_io):
    from ragwrench.port import status
    path,bundle,_=skip_mix
    rows=expected_rows(path,bundle)
    original=status.read_json_snapshot
    data,calls=path.read_bytes(),[]
    def changed_after_read(p):
        calls.append(p); result=original(p); path.write_bytes(data+b' '); return result
    monkeypatch.setattr(status,'read_json_snapshot',changed_after_read)
    token=import_status(path,bundle,list_bundle_chunks=True,bundle_chunk_page_size=1)['bundle_progress']['next_bundle_chunk_cursor']
    assert calls==[path] and decode_cursor(token)['state']==hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(status,'read_json_snapshot',original)
    path.write_bytes(data)
    copy=tmp_path/'copy.json'; copy.write_bytes(data)
    other=tmp_path/'other'; shutil.copytree(bundle,other)
    full=import_status(copy,other)
    assert invoke(copy,other,capsys,1)['bundle_progress']['next_bundle_chunk_cursor']==token
    token=assert_page(invoke(copy,other,capsys,4,token),full,rows[1:5])
    assert assert_page(invoke(copy,other,capsys,10,token),full,rows[5:]) is None
    for local_status in (None,'completed'):
        with pytest.raises(PortError,match='Invalid chunk continuation'):
            import_status(path,bundle,document='a',list_chunks=True,page_size=1,cursor=token,chunk_status=local_status)
    with pytest.raises(PortError,match='Invalid skip continuation'):
        import_status(path,bundle,list_skipped_chunks=True,skip_page_size=1,skip_cursor=token)
    with pytest.raises(PortError,match='Invalid document continuation'):
        import_status(path,bundle,pending_stage='enabled_flag',document_page_size=1,document_cursor=token)
    for initial_status in STATUSES:
        filtered=[r for r in rows if initial_status is None or r['status']==initial_status]
        if len(filtered)<2: continue
        bound=invoke(path,bundle,capsys,1,status=initial_status)['bundle_progress']['next_bundle_chunk_cursor']
        for changed_status in STATUSES:
            if changed_status==initial_status: continue
            with pytest.raises(PortError,match='Invalid bundle chunk continuation'):
                import_status(path,bundle,list_bundle_chunks=True,bundle_chunk_page_size=1,
                              bundle_chunk_cursor=bound,bundle_chunk_status=changed_status)


@pytest.mark.parametrize("mode", ["list", "page", "cursor"])
@pytest.mark.parametrize("case,error", [("schema", "Bundle schema violation"),
    ("relationship", "Chunk references an unknown document"), ("content-hash", "Chunk content_sha256 mismatch"),
    ("counts", "Manifest counts do not match JSONL records")])
def test_bundle_chunks_validates_non_skipped_off_page_bundle(skip_mix, capsys, request, monkeypatch, mode, case, error):
    path, bundle, state = skip_mix
    m, _, chunks = load_bundle(bundle)
    # Last document e has no skips, and cannot appear in this output.
    if case == "schema": chunks[-1]["content"] = 42
    elif case == "relationship": chunks[-1]["doc_id"] = "private-unknown"
    elif case == "content-hash": chunks[-1]["content"] = "private changed"
    else: m["counts"]["chunks"] += 1
    blob = "".join(json.dumps(c) + "\n" for c in chunks).encode()
    (bundle / "chunks.jsonl").write_bytes(blob)
    m["sha256"]["chunks.jsonl"] = hashlib.sha256(blob).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(m))
    state["identity"]["bundle"] = fingerprint(m)
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    monkeypatch.setattr("ragwrench.port.status.make_plan", lambda *a, **k: pytest.fail("invalid bundle reached preflight"))
    options = {"list_bundle_chunks": True, "bundle_chunk_status": "skipped"}
    args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-bundle-chunks", "--bundle-chunk-status", "skipped"]
    if mode != "list":
        options["bundle_chunk_page_size"] = 1; args += ["--bundle-chunk-page-size", "1"]
    if mode == "cursor":
        options["bundle_chunk_cursor"] = "private-invalid"; args += ["--bundle-chunk-cursor", "private-invalid"]
    before = files_snapshot(path.parent)
    with pytest.raises(PortError, match=error): import_status(path, bundle, **options)
    assert main(args) == 1
    out = capsys.readouterr()
    assert not out.out and error in out.err and "private" not in out.err and "cursor" not in out.err
    assert files_snapshot(path.parent) == before


@pytest.mark.parametrize("mode", ["list", "page", "cursor"])
@pytest.mark.parametrize("case", ["off-page-skipped", "off-page-nonskipped", "incomplete", "pending-skip"])
def test_bundle_chunks_invalid_off_page_mapping(skip_mix, capsys, request, mode, case):
    path, bundle, state = skip_mix
    if case == "off-page-skipped": state["documents"]["c"]["chunks"]["c1"] = "tc1"
    elif case == "off-page-nonskipped": state["documents"]["e"]["chunks"]["foreign"] = "tf"
    elif case == "incomplete": state["documents"]["e"]["chunks"] = {}
    else: state["documents"]["a"]["pending"]["source"] = "c3"
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    options = dict(list_bundle_chunks=True, bundle_chunk_status="skipped", bundle_chunk_page_size=None if mode == "list" else 1,
                   bundle_chunk_cursor="private-invalid" if mode == "cursor" else None)
    with pytest.raises(PortError, match="Invalid import state"): import_status(path, bundle, **options)
    args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-bundle-chunks", "--bundle-chunk-status", "skipped"]
    if mode != "list": args += ["--bundle-chunk-page-size", "1"]
    if mode == "cursor": args += ["--bundle-chunk-cursor", "private-invalid"]
    assert main(args) == 1
    out = capsys.readouterr()
    assert not out.out and "Invalid import state" in out.err and "private" not in out.err


@pytest.mark.parametrize("mode", ["list", "page"])
@pytest.mark.parametrize("kind", ["missing", "malformed", "symlink", "directory", "oversized"])
def test_bundle_chunks_state_file_errors(tmp_path, bundle, capsys, forbid_io, kind, mode):
    path = tmp_path / "missing" / "state.json"
    if kind != "missing": path.parent.mkdir()
    if kind == "malformed": path.write_text('{"secret":"private"')
    elif kind == "symlink": path.symlink_to(bundle / "manifest.json")
    elif kind == "directory": path.mkdir()
    elif kind == "oversized":
        with path.open("wb") as stream: stream.truncate(256 * 1024 * 1024 + 1)
    before = set(tmp_path.rglob("*"))
    args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-bundle-chunks", "--bundle-chunk-status", "skipped"]
    if mode == "page": args += ["--bundle-chunk-page-size", "1"]
    assert main(args) == 1
    out = capsys.readouterr()
    assert not out.out and out.err and "private" not in out.err and "Traceback" not in out.err
    assert set(tmp_path.rglob("*")) == before
