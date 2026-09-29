"""
CERT r4.2 Insider-Threat Preprocessing Pipeline
================================================
Faithfully reconstructed from the original dataClean.py (ITDBERT project,
github.com/cgly/ITDBERT).

Usage
-----
    python dataClean.py --input /path/to/r4.2 --output /path/to/output

Expected input structure
------------------------
    r4.2/
    ├── logon.csv
    ├── http.csv
    ├── email.csv
    ├── file.csv
    ├── device.csv
    └── answers/
        ├── insiders.csv
        ├── r4.2-1/
        │   ├── r4.2-1-AAM0658.csv
        │   └── ...
        ├── r4.2-2/
        │   └── ...
        └── r4.2-3/
            └── ...

Pipeline
--------
 1.  Load insider list from answers/insiders.csv
 2.  Encode malicious-user log events          → mal_events_sorted.csv
 3.  Merge into per-(user, day) sequences      → train_annormal.csv
 4.  Reduce HTTP events                        → Htrain_annormal.csv
 5.  Extract answer-file events                → milres.csv
 6.  Convert answer events to numeric ids      → milnum.csv
 7.  Merge answer events by (user, day)        → MergeMilnum.csv
 8.  Annotate malicious events in sequences    → 42HAT.csv
 9.  Generate MIL weight vectors               → selectMilItem_out.csv
10.  Encode normal-user log events             → normal_events_sorted.csv
11.  Merge normal sequences by (user, day)     → train_normal.csv
12.  Reduce HTTP events (normal)               → Htrain_normal.csv
13.  Build final ITDBERT train/test CSV files  → THNS24_2010.csv
                                               → THNS24_2011.csv

NEW in this version (dataclean_meta.py)
---------------------------------------
  * Every final CSV now has a companion  <name>_meta.csv  with the SAME row
    order:  user, date, label, user_scenario   (scenario 1/2/3 read from the
    answers/ file names, 0 = user not in any scenario).
    The existing CSV format is unchanged, so all current scripts still work.
  * --reduced-only : build only the reduced (insiders-only) datasets and skip
    the heavy normal-user steps 10-13 (they are not needed for the reduced set).
  * Log filtering is now vectorised with pandas (same result, much faster).

Final CSV format (THNS24_*.csv)
--------------------------------
    label, ev1, ev2, …, evN   (variable length, no padding; cap at 512)
    label  : 0 = normal user,  1 = insider
    events : minute-slot num_id ÷ 60  (hour-slot, 0–168)
    2010   : training set   (all normal days + insider days in 2010)
    2011   : test set       (all normal days + insider days in 2011)

"""

import argparse
import csv
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

act_dict = {
    'Logon':      1,
    'Logoff':     2,
    'http':       3,
    'email':      4,
    'file':       5,
    'Connect':    6,
    'Disconnect': 7,
}

# The five raw log files expected inside the r4.2 root folder
LOG_FILES = {
    'logon':  'logon.csv',
    'device': 'device.csv',
    'email':  'email.csv',
    'file':   'file.csv',
    'http':   'http.csv',
}

# Answer scenario sub-directories to walk
ANSWER_SUBDIRS = ['r4.2-1', 'r4.2-2', 'r4.2-3']


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def time_convert(timestr, action):
    """
    Map an HH:MM time string and an action-category integer (1-7) to a
    unique minute-slot integer.

    Each category occupies a contiguous 1440-slot window:
        Logon      →    1 – 1 440
        Logoff     → 1441 – 2 880
        http       → 2881 – 4 320
        email      → 4321 – 5 760
        file       → 5761 – 7 200
        Connect    → 7201 – 8 640
        Disconnect → 8641 – 10 080
    """
    h, m = timestr.split(":")[:2]
    offset = 60 * int(h) + int(m)
    return (action - 1) * 1440 + offset + 1


def isHTTp(num):
    """Return True when the slot belongs to the HTTP action window (2881–4320)."""
    return 2881 <= int(num) <= 4320


def _action_from_row(filetype, row):
    """Infer the action name from a log-file type and a CSV row dict."""
    if filetype in ('logon', 'device'):
        return row.get('activity', '')
    return filetype   # 'http', 'email', 'file'


def _reformat_date(raw_date):
    """
    Convert 'MM/DD/YYYY HH:MM:SS' → 'YYYY/MM/DD HH:MM:SS'.
    Mirrors the original expression: date[6:10]+date[5]+date[:5]+date[10:]
    """
    return raw_date[6:10] + raw_date[5] + raw_date[:5] + raw_date[10:]


def _read_sorted_csv(path):
    with open(path, newline='') as f:
        return [r for r in csv.reader(f) if any(r)]


def _write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Original functions
# ─────────────────────────────────────────────────────────────────────────────

def ShowFirst5Col(path, isshowFirst5):
    pd.set_option('display.max_columns', None)
    data = pd.read_csv(path, nrows=10) if isshowFirst5 else pd.read_csv(path)
    print(data)


def saveDf2csv(path, newPath):
    """Drop columns unused by the pipeline and resave as CSV."""
    data = pd.read_csv(path)
    drop = [c for c in ('id', 'pc', 'content', 'filename') if c in data.columns]
    data.drop(drop, axis=1).to_csv(newPath, index=False)


def delMulseq(MilFile_Path, oriFile_Path, tarFile_Path):
    """
    Remove every row whose first column (user) appears in the insider list.
    Produces a clean corpus of normal-user events only.
    """
    with open(MilFile_Path) as f:
        MilUser = {row[0] for row in csv.reader(f)}

    count = 0
    with open(oriFile_Path) as r, open(tarFile_Path, 'w', newline='') as w:
        writer = csv.writer(w)
        for line in csv.reader(r):
            if line[0] not in MilUser:
                writer.writerow(line)
            else:
                count += 1
    print(f"Removed {count} rows belonging to {len(MilUser)} malicious users.")


def httpreduce(oriFile_Path, tarFile_Path, timestamp, mul_sum):
    """
    Sub-sample HTTP events: keep one event at most every `timestamp` minutes.
    HTTP slots occupy the range 2881–4320.
    """
    with open(oriFile_Path) as read, open(tarFile_Path, 'w', newline='') as write:
        writer = csv.writer(write)
        for line in csv.reader(read):
            line = [i for i in line if i != '']
            new_seq  = []
            ishttpStart = 1
            cur_http    = 0
            for j, item in enumerate(line):
                if j < 3:
                    new_seq.append(item)
                    continue
                item = int(item)
                if 2881 <= item <= 4320:
                    if ishttpStart:
                        cur_http    = item
                        ishttpStart = 0
                        new_seq.append(item)
                    else:
                        if item - cur_http >= timestamp:
                            cur_http += timestamp
                            new_seq.append(cur_http)
                else:
                    prev = int(line[j - 1]) if j > 2 else 0
                    if 2881 <= prev <= 4320 and prev != cur_http:
                        new_seq.append(prev)
                    ishttpStart = 1
                    new_seq.append(item)
            writer.writerow(new_seq)


def recordMil(answers_dir, target_path):
    """
    Walk every answer sub-folder (r4.2-1, r4.2-2, r4.2-3), parse each
    malicious-event CSV and write rows of:
        [date YYYY/MM/DD HH:MM:SS, user, action_name]
    """
    milRes = []
    for sub in ANSWER_SUBDIRS:
        sub_path = os.path.join(answers_dir, sub)
        if not os.path.isdir(sub_path):
            print(f"  [skip] {sub_path} not found")
            continue
        for fname in sorted(os.listdir(sub_path)):
            if not fname.endswith('.csv'):
                continue
            with open(os.path.join(sub_path, fname)) as f:
                for row in csv.reader(f):
                    if len(row) < 6:
                        continue
                    filetype = row[0]
                    raw_date = row[2]   # MM/DD/YYYY HH:MM:SS
                    user     = row[3]
                    try:
                        date = _reformat_date(raw_date)
                    except IndexError:
                        continue
                    # Logon/device rows carry the activity name in column 5
                    action = row[5] if filetype in ('device', 'logon') else filetype
                    milRes.append([date, user, action])

    _write_csv(target_path, milRes)
    print(f"recordMil: {len(milRes)} events → {target_path}")
    return milRes


def milAct2Num(file_path, target_path):
    """
    Convert [date YYYY/MM/DD HH:MM:SS, user, action_name]
         → [date YYYY/MM/DD,           user, num_id]
    using time_convert.
    Returns also a parallel list of raw_offsets (h*60+m) for correct sorting.
    """
    res = []
    dts = []
    with open(file_path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            date_str, user, action_name = row[0], row[1], row[2]
            if action_name not in act_dict:
                continue
            try:
                dt        = datetime.strptime(date_str[:19], '%Y/%m/%d %H:%M:%S')
                date_part = dt.strftime('%Y/%m/%d')
                time_part = dt.strftime('%H:%M')
                num_id    = time_convert(time_part, act_dict[action_name])
                res.append([date_part, user, num_id])
                dts.append(dt)
            except Exception:
                continue

    _write_csv(target_path, res)
    print(f"milAct2Num: {len(res)} events → {target_path}")
    return dts


def mergeUser(file_path, target_path):
    """
    Group consecutive rows sharing the same (date, user) into one sequence:
        [date, user, event_1, event_2, …]

    Input must be sorted by (user, date, num_id).
    """
    with open(file_path) as f, open(target_path, 'w', newline='') as w:
        writer   = csv.writer(w)
        cur_date = cur_name = None
        action   = []
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            if cur_date is None:
                cur_date, cur_name = row[0], row[1]
                action = [cur_date, cur_name, row[2]]
            elif cur_date == row[0] and cur_name == row[1]:
                action.append(row[2])
            else:
                writer.writerow(action)
                cur_date, cur_name = row[0], row[1]
                action = [cur_date, cur_name, row[2]]
        if action:
            writer.writerow(action)

    print(f"mergeUser → {target_path}")


def tag_compareData(mil_path, day_path, save_path):
    """
    For each (date, user) day-sequence, mark every event that appears in the
    corresponding answer-sequence with a '-' prefix (negative = malicious).

    mil_path : MergeMilnum.csv      — answer sequences  [date, user, ev, …]
    day_path : Htrain_annormal.csv  — log sequences     [date, user, ev, …]
    save_path: 42HAT.csv            — annotated output
    """
    # Build lookup: (date, user) → list of malicious num_ids
    mil_lookup: dict = {}
    with open(mil_path) as f:
        for row in csv.reader(f):
            row = [i for i in row if i != '']
            if len(row) < 3:
                continue
            key = (row[0], row[1])
            mil_lookup.setdefault(key, []).extend(int(x) for x in row[2:])

    count  = 0
    result = []
    with open(day_path) as f:
        for row in csv.reader(f):
            row = [i for i in row if i != '']
            if len(row) < 3:
                continue
            key      = (row[0], row[1])
            day_copy = list(row)

            if key in mil_lookup:
                for mil_ev in mil_lookup[key]:
                    cur_num  = int(mil_ev)
                    day_nums = [abs(int(x)) for x in day_copy[2:]]

                    if isHTTp(cur_num):
                        # Map to the nearest HTTP slot in the day sequence
                        diffs = [abs(n - cur_num) for n in day_nums]
                        ind   = diffs.index(min(diffs)) + 2
                    else:
                        if cur_num not in day_nums:
                            continue
                        ind = day_nums.index(cur_num) + 2

                    day_copy[ind] = '-' + str(abs(int(day_copy[ind])))
                    count += 1

            result.append(day_copy)

    _write_csv(save_path, result)
    print(f"tag_compareData: {count} events tagged across {len(result)} sequences → {save_path}")


def selectMilItem(file_path, tarFilePath, skip_year='2010'):
    """
    Convert tagged sequences to MIL weight vectors (fixed length 67).
    Malicious event (negative value) → 0.2 ; normal event → 0.01.
    Rows whose date starts with `skip_year` are excluded.
    """
    with open(file_path) as r, open(tarFilePath, 'w', newline='') as w:
        writer = csv.writer(w)
        for line in csv.reader(r):
            line = [i for i in line if i != '']
            if not line:
                continue
            if str(line[0]).startswith(skip_year):   # FIX: was line[1]
                continue
            wList = []
            for idx, val in enumerate(line):
                if idx < 2:
                    wList.append(val)
                    continue
                wList.append(0.2 if int(val) < 0 else 0.01)
            # Pad or truncate to exactly 67 columns
            new_seq = [wList[i] if i < len(wList) else 0.01 for i in range(67)]
            writer.writerow(new_seq)

    print(f"selectMilItem → {tarFilePath}")


def getMilData(file_path, tarFilePath, skip_year='2010'):
    """
    Convert HTTP-reduced sequences to model-ready input.
    Each minute-slot value is divided by 60 to obtain an hour-slot.
    Rows whose date starts with `skip_year` are excluded.
    The first output column is always 1 (insider label).
    """
    with open(file_path) as r, open(tarFilePath, 'w', newline='') as w:
        writer = csv.writer(w)
        for line in csv.reader(r):
            line = [i for i in line if i != '']
            if len(line) < 3:
                continue
            if str(line[0]).startswith(skip_year):
                continue
            wList = [1]   # insider label
            for idx, val in enumerate(line):
                if idx < 2:
                    continue
                if idx == 2:
                    wList.append(1)
                    continue
                wList.append(int(int(val) / 60))
            writer.writerow(wList)

    print(f"getMilData → {tarFilePath}")


# ─────────────────────────────────────────────────────────────────────────────
# New helpers — read log CSVs and encode events
# ─────────────────────────────────────────────────────────────────────────────

def _count_lines(fpath):
    """Count lines in a file efficiently (used to size the tqdm total)."""
    count = 0
    with open(fpath, 'rb') as f:
        while chunk := f.read(1 << 20):
            count += chunk.count(b'\n')
    return max(count - 1, 0)              # subtract header line


def _encode_log_events(data_dir, user_filter, keep_if_in_filter):
    """
    Read all 5 log CSVs in chunks (memory-efficient), filter by user
    membership, and return a sorted list of [date YYYY/MM/DD, user, num_id].

    keep_if_in_filter=True  → keep rows whose user IS in user_filter
    keep_if_in_filter=False → keep rows whose user is NOT in user_filter

    Uses chunksize=100 000 rows so even a 14 GB CSV never fully loads into
    RAM.
    """
    CHUNKSIZE = 100_000
    DROP_COLS = {'id', 'pc', 'content', 'filename'}

    events = []
    for filetype, fname in LOG_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath} not found")
            continue

        print(f"  {fname} — counting rows …", end='\r')
        n_rows  = _count_lines(fpath)
        n_chunks = (n_rows + CHUNKSIZE - 1) // CHUNKSIZE

        reader = pd.read_csv(
            fpath,
            chunksize=CHUNKSIZE,
            usecols=lambda c: c not in DROP_COLS,
            low_memory=False,
        )

        with tqdm(
            reader,
            total=n_chunks,
            desc=f"  {fname}",
            unit="chunk",
            unit_scale=False,
            postfix={"rows": f"{n_rows:,}"},
        ) as bar:
            for chunk in bar:
                # SPEED-UP: same selection as the old per-row test, but vectorised
                mask  = chunk['user'].astype(str).isin(user_filter)
                chunk = chunk[mask] if keep_if_in_filter else chunk[~mask]
                for _, row in chunk.iterrows():
                    user      = str(row['user'])
                    action_name = _action_from_row(filetype, row)
                    if action_name not in act_dict:
                        continue
                    raw_date = str(row['date'])   # MM/DD/YYYY HH:MM:SS
                    try:
                        dt = datetime.strptime(raw_date[:19], '%m/%d/%Y %H:%M:%S')
                        date_part = dt.strftime('%Y/%m/%d')
                        time_part = dt.strftime('%H:%M')
                        num_id    = time_convert(time_part, act_dict[action_name])
                        events.append([date_part, user, num_id, dt])
                    except Exception:
                        continue

    # Sort by (user, actual datetime)
    events.sort(key=lambda x: (x[1], x[3]))
    return [[e[0], e[1], e[2]] for e in events]


def buildMaliciousSequences(data_dir, malicious_users, out_path):
    """Extract and encode log events for malicious users only."""
    events = _encode_log_events(data_dir, malicious_users, keep_if_in_filter=True)
    _write_csv(out_path, events)
    print(f"buildMaliciousSequences: {len(events)} events → {out_path}")


def buildNormalSequences(data_dir, malicious_users, out_path):
    """Extract and encode log events for normal users (no malicious user included)."""
    events = _encode_log_events(data_dir, malicious_users, keep_if_in_filter=False)
    _write_csv(out_path, events)
    print(f"buildNormalSequences: {len(events)} events → {out_path}")


def loadInsiders(insiders_csv, dataset='4.2'):
    """Return the set of malicious user IDs for the given dataset version."""
    users = set()
    with open(insiders_csv) as f:
        for row in csv.DictReader(f):
            if row.get('dataset', '').strip() == dataset:
                users.add(row['user'].strip())
    return users


def build_model_csv(sequences_path, label, keep_year, seq_len=512,
                    malicious_days=None, keep_malicious=True, meta_out=None):
    """
    Convert sequences [date, user, ev1, ev2, …] into ITDBERT model rows:
        [label, ev1÷60, ev2÷60, …]

    Parameters
    ----------
    sequences_path : path to an HTTP-reduced sequence CSV
    label          : 0 (normal) or 1 (insider)
    keep_year      : '2010' or '2011' — only rows from this year are kept
    seq_len        : hard truncation cap (default 512)
    malicious_days : optional set of (date, user) tuples identifying truly
                     malicious days (from MergeMilnum / answer files).
                     When provided, each row is kept only if its (date, user)
                     membership matches `keep_malicious`:
                       keep_malicious=True  → keep only malicious-day rows
                       keep_malicious=False → keep only non-malicious-day rows
                     When None, all rows pass through regardless.

    meta_out : optional list; when given, [user, date] is appended for every
               kept row, in the same order as the returned rows.

    Returns a list of rows ready to be written to the final CSV.
    """
    rows = []
    with open(sequences_path) as f:
        for line in csv.reader(f):
            line = [i for i in line if i != '']
            if len(line) < 3:
                continue
            if not line[0].startswith(keep_year):
                continue
            if malicious_days is not None:
                is_malicious = (line[0], line[1]) in malicious_days
                if is_malicious != keep_malicious:
                    continue
            # Convert minute-slots to hour-slots (÷60), drop date & user cols
            events = [int(int(v) / 60) for v in line[2:]]
            if seq_len is not None:
                events = events[:seq_len]
            rows.append([label] + events)
            if meta_out is not None:
                meta_out.append([line[1], line[0]])      # [user, date]
    return rows


def load_user_scenarios(answers_dir):
    """
    Map user -> scenario number from the answer file names
    (answers/r4.2-1/r4.2-1-AAM0658.csv  ->  {'AAM0658': 1}).
    """
    scen = {}
    for sub in ANSWER_SUBDIRS:
        sub_path = os.path.join(answers_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        number = int(sub.rsplit('-', 1)[1])
        for fname in os.listdir(sub_path):
            if fname.endswith('.csv'):
                scen[fname[:-4].rsplit('-', 1)[1]] = number
    return scen


def _write_meta(path, rows, metas, scenarios):
    """Write <user, date, label, user_scenario> aligned row-by-row with `rows`."""
    assert len(rows) == len(metas), "meta / rows length mismatch"
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['user', 'date', 'label', 'user_scenario'])
        for row, (user, date) in zip(rows, metas):
            w.writerow([user, date, row[0], scenarios.get(user, 0)])


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(r42_dir, output_dir='./output', reduced_only=False):
    """
    End-to-end CERT r4.2 preprocessing.

    Parameters
    ----------
    r42_dir    : root folder containing the 5 log CSVs and the answers/ sub-folder
    output_dir : destination for all intermediate and final files
    """
    os.makedirs(output_dir, exist_ok=True)
    answers_dir = os.path.join(r42_dir, 'answers')

    def p(name):
        return os.path.join(output_dir, name)

    # ── 1. Load insider list ──────────────────────────────────────────────
    print("\n── Step  1 / 13 : Load insider list ─────────────────────────────────")
    malicious_users = loadInsiders(os.path.join(answers_dir, 'insiders.csv'))
    print(f"   {len(malicious_users)} malicious users identified.")

    # ── 2. Encode malicious-user log events ───────────────────────────────
    print("\n── Step  2 / 13 : Encode malicious-user log events ──────────────────")
    mal_sorted = p('mal_events_sorted.csv')
    buildMaliciousSequences(r42_dir, malicious_users, mal_sorted)

    # ── 3. Merge into per-(user, day) sequences ───────────────────────────
    print("\n── Step  3 / 13 : Merge into per-user-per-day sequences ─────────────")
    train_ann = p('train_annormal.csv')
    mergeUser(mal_sorted, train_ann)

    # ── 4. HTTP reduction ─────────────────────────────────────────────────
    print("\n── Step  4 / 13 : Reduce HTTP events (window = 30 min) ─────────────")
    Htrain_ann = p('Htrain_annormal.csv')
    httpreduce(train_ann, Htrain_ann, timestamp=30, mul_sum=0)

    # ── 5. Parse answer files ─────────────────────────────────────────────
    print("\n── Step  5 / 13 : Parse answer files ────────────────────────────────")
    milres = p('milres.csv')
    recordMil(answers_dir, milres)

    # ── 6. Convert answer events to numeric ids ───────────────────────────
    print("\n── Step  6 / 13 : Convert answer events to numeric ids ──────────────")
    milnum = p('milnum.csv')
    dts = milAct2Num(milres, milnum)

    # Sort milnum by (user, actual datetime) — chronologically exact
    rows = _read_sorted_csv(milnum)
    combined = list(zip(rows, dts))
    combined.sort(key=lambda t: (t[0][1], t[1]) if len(t[0]) >= 3 else (t[0],))
    rows = [r for r, _ in combined]
    _write_csv(milnum, rows)

    # ── 7. Merge answer sequences by (user, day) ──────────────────────────
    print("\n── Step  7 / 13 : Merge answer sequences ────────────────────────────")
    merge_mil = p('MergeMilnum.csv')
    mergeUser(milnum, merge_mil)

    # ── 8. Annotate malicious events ──────────────────────────────────────
    print("\n── Step  8 / 13 : Annotate malicious events ─────────────────────────")
    tagged = p('42HAT.csv')
    tag_compareData(merge_mil, Htrain_ann, tagged)

    # ── 9. MIL weight vectors ─────────────────────────────────────────────
    print("\n── Step  9 / 13 : Generate MIL weight vectors ───────────────────────")
    selectMilItem(tagged, p('selectMilItem_out.csv'))

    # ── 10-12. Normal-user sequences (skipped with --reduced-only) ─────────
    Htrain_normal = p('Htrain_normal.csv')
    if not reduced_only:
        print("\n── Step 10 / 13 : Encode normal-user log events ─────────────────────")
        normal_sorted = p('normal_events_sorted.csv')
        buildNormalSequences(r42_dir, malicious_users, normal_sorted)

        print("\n── Step 11 / 13 : Merge normal sequences ────────────────────────────")
        train_normal = p('train_normal.csv')
        mergeUser(normal_sorted, train_normal)

        print("\n── Step 12 / 13 : Reduce HTTP events — normal users ─────────────────")
        httpreduce(train_normal, Htrain_normal, timestamp=30, mul_sum=0)
    else:
        print("\n── Steps 10-12 skipped (--reduced-only) ─────────────────────────────")

    # Build the set of (date, user) pairs that are truly malicious days
    # (i.e. appear in the ground-truth answer sequences MergeMilnum.csv).
    malicious_days = set()
    with open(merge_mil) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                malicious_days.add((row[0], row[1]))   # (date YYYY/MM/DD, user)
    print(f"   {len(malicious_days)} malicious (user, day) pairs from answer files.")

    scenarios = load_user_scenarios(answers_dir)
    print(f"   scenario known for {len(scenarios)} users.")

    def _normal(label, year, keep_malicious, meta):
        """Normal-user rows; empty (with a warning) when --reduced-only."""
        if reduced_only:
            return []
        return build_model_csv(Htrain_normal, label=label, keep_year=year,
                               malicious_days=malicious_days,
                               keep_malicious=keep_malicious, meta_out=meta)

    if reduced_only:
        stray = {u for _, u in malicious_days} - set(malicious_users)
        if stray:
            print(f"   WARNING: {len(stray)} answer-file users are not in insiders.csv "
                  f"({sorted(stray)[:5]} ...). Their attack days would be added as "
                  f"label=1 by the full mode; rerun without --reduced-only to include them.")

    # ── 13. Assemble THNS24_2010.csv and THNS24_2011.csv (full mode only) ──
    if not reduced_only:
        print("\n── Step 13 / 13 : Assemble final ITDBERT train / test CSV files ─────")
        for year in ('2010', '2011'):
            m_norm, m_att, m_ins, m_non = [], [], [], []
            normal_rows        = _normal(0, year, False, m_norm)
            normal_attack_rows = _normal(1, year, True,  m_att)
            insider_rows  = build_model_csv(Htrain_ann, label=1, keep_year=year,
                                            malicious_days=malicious_days, meta_out=m_ins)
            non_mal_rows  = build_model_csv(Htrain_ann, label=0, keep_year=year,
                                            malicious_days=malicious_days,
                                            keep_malicious=False, meta_out=m_non)
            all_rows  = normal_rows + normal_attack_rows + insider_rows + non_mal_rows
            all_meta  = m_norm + m_att + m_ins + m_non
            out_path  = p(f'THNS24_{year}.csv')
            _write_csv(out_path, all_rows)
            _write_meta(p(f'THNS24_{year}_meta.csv'), all_rows, all_meta, scenarios)
            print(f"  THNS24_{year}.csv : {len(normal_rows)} normal-user + "
                  f"{len(normal_attack_rows)} non-insider-attack (label=1) + "
                  f"{len(insider_rows)} insider-day + "
                  f"{len(non_mal_rows)} insider-normal-day"
                  f" = {len(all_rows)} sequences → {out_path}")

    # ── 14. Reduced datasets (insider sequences only) ──────────────────────
    print("\n── Step 14 / 14 : Assemble reduced datasets (insiders only) ─────────")
    for year in ('2010', '2011'):
        m_ins, m_non, m_att = [], [], []
        insider_rows  = build_model_csv(Htrain_ann, label=1, keep_year=year,
                                        malicious_days=malicious_days, meta_out=m_ins)
        non_mal_rows  = build_model_csv(Htrain_ann, label=0, keep_year=year,
                                        malicious_days=malicious_days,
                                        keep_malicious=False, meta_out=m_non)
        # Non-insider users present in attack scenarios → label 1
        normal_attack_rows = _normal(1, year, True, m_att)
        all_rows  = insider_rows + non_mal_rows + normal_attack_rows
        all_meta  = m_ins + m_non + m_att
        out_path  = p(f'reduced_THNS24_{year}.csv')
        _write_csv(out_path, all_rows)
        _write_meta(p(f'reduced_THNS24_{year}_meta.csv'), all_rows, all_meta, scenarios)
        print(f"  reduced_THNS24_{year}.csv : "
              f"{len(insider_rows)} insider-day (label=1) + "
              f"{len(normal_attack_rows)} non-insider-attack (label=1) + "
              f"{len(non_mal_rows)} insider-normal-day (label=0) "
              f"= {len(all_rows)} sequences → {out_path}")

    print(f"""
══════════════════════════════════════════════════════════════════
  Pipeline complete.
  Output directory: {output_dir}
══════════════════════════════════════════════════════════════════
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CERT r4.2 insider-threat preprocessing pipeline (ITDBERT).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python dataClean.py --input ./r4.2 --output ./output

The r4.2 folder must contain:
    logon.csv  http.csv  email.csv  file.csv  device.csv
    answers/insiders.csv
    answers/r4.2-1/   answers/r4.2-2/   answers/r4.2-3/
        """
    )
    parser.add_argument(
        '--input', '-i', required=True,
    )
    parser.add_argument(
        '--output', '-o', default='./output',
    )
    parser.add_argument(
        '--reduced-only', action='store_true',
        help='Build only reduced_THNS24_*.csv (insiders only); skips normal-user steps 10-13.',
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, reduced_only=args.reduced_only)