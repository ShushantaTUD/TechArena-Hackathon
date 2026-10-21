# TechArena 2026 – Phase 1: Click Prediction

## Task

Phase 1 is a click-probability prediction task on synthetic recommendation data. No real users or products are involved. Your task is to predict a click probability for every row in `evaluation.csv`.

- An **event** is one recommendation opportunity: one user is shown **10 candidate items** at one moment.
- A **row** in the CSV files is one candidate (an *impression*): one item shown to one user in one event.
- In the training data the target `clicked` is 1 if that candidate was clicked and 0 otherwise.
- An event has **at most one clicked candidate**. Events with no click are also possible: all 10 of their rows have `clicked = 0`.

For **every row of `evaluation.csv`**, predict the probability that the candidate was clicked. Do not submit an extra row or
value for the "no click" outcome.

## Files

| File | Content | Data rows (header excluded) |
|---|---|---|
| `train.csv` | Labelled candidates (includes `clicked`), 686,824 events | 6,868,240 |
| `evaluation.csv` | Candidates to predict, no labels, 52,939 events (one per user) | 529,390 |
| `example_submission.csv` | Format example: one line per evaluation row, all with the same constant value | 529,390 |
| `README.txt` | Short description shipped with the data | – |

`train.csv` is about 7.3 GB. Consider streaming or chunked processing. Labels for the evaluation rows are not provided.

## Columns

`evaluation.csv` has the same columns as `train.csv` except `clicked`.

| Column | Meaning |
|---|---|
| `rowid` | Identifier of the row (one candidate in one event). The key of your submission. |
| `event_id` | Identifier of the event. All 10 rows of an event share user, timestamp and history. |
| `user_id` | The user. |
| `item_id` | The candidate item shown in this row. |
| `timestamp` | Event time, Unix seconds, UTC (whole minutes). |
| `time_bucket` | Six-hour block of the UTC day: 0 = 00:00–05:59, 1 = 06:00–11:59, 2 = 12:00–17:59, 3 = 18:00–23:59. |
| `day_type` | 0 = Monday–Friday, 1 = Saturday or Sunday (UTC date). |
| `history_item_ids` | JSON array of items this user clicked before the event. |
| `history_timestamps` | JSON array of the Unix-second times of those clicks. |
| `clicked` | **Training only.** 1 = clicked, 0 = not clicked. |

- `rowid`, `event_id`, `user_id` and `item_id` are **opaque identifiers**. Treat them as text, not numbers, and do not try to
  interpret them. `rowid` and `event_id` are random and carry no information; use them as keys only.
- The history columns hold JSON arrays inside quoted CSV fields. Read the files with a CSV parser, then parse these fields
  with a JSON parser; do not split lines on commas.

## History and time split

- A history contains previous **clicks** only (not all earlier impressions): the latest **20 at most**, strictly before the
  current event, ordered **oldest to newest**.
- `history_item_ids` and `history_timestamps` are aligned element by element and have the same length. An empty history is `[]`.
  The same item can appear several times.
- The simulated timeline starts before the training period (a *warm-up* period, before 2025-01-20 UTC). Its impression rows are
  **not** included, but its clicks can appear in histories. A user can have a non-empty history without any row in `train.csv`.
- **Training** covers events from 2025-01-20 00:00 UTC up to, not including, 2025-01-27 00:00 UTC (the cutoff).
- **Evaluation** contains, for each user, the **first event at or after the cutoff**, with all its 10 candidates. Later events of
  these users are not included.
- Some evaluation users do not appear in `train.csv`, and some evaluation events have an empty history. Make sure your
  predictions cover these cases too.

**Local validation.** You have no evaluation labels, so validate on a later part of the training period: fit on earlier events,
test on later ones, and keep all rows of an event on the same side of the split. Compute features and aggregate statistics (for
example click rates per user or item) using labels from the fitting part only, never from the events you are testing on.
A later training window does not exactly reproduce the official setup, which uses one first event per user.

## Submission format

A UTF-8 CSV file with exactly this header (case-sensitive, this order):

```
rowid,predicted_probability
```

Format illustration only – a real submission has one line for **every** evaluation row:

```
rowid,predicted_probability
row_aaaa0000,0.0629
row_bbbb1111,0.1200
row_cccc2222,0.0003
```

Requirements:

- Every `rowid` of `evaluation.csv` appears **exactly once**: no missing, duplicate or unknown rowids.
- Exactly the two columns above and no extra index column. Keep each `rowid` exactly as given (no trimming or case changes).
- Each prediction is a finite number in **[0, 1]**. Plain decimals (`0.25`, `.5`, `1`) and scientific notation (`2e-3`) are accepted.
  Empty values, text, `nan` and `inf` are rejected.
- Predictions are matched to rows **by `rowid`**, so row order does not matter. Keeping the order of `example_submission.csv`
  is still convenient.
- The probabilities of an event do **not** have to sum to one.

Upload the finished file through the competition's submission page, then check its processing status in your submission
history. If an upload is interrupted, check the history before uploading again.

## Evaluation

Submissions are ranked by **ROC AUC**; higher is better. AUC measures how well your predictions rank clicked rows above
non-clicked rows: it is the probability that a randomly chosen clicked row gets a higher prediction than a randomly chosen
non-clicked row (ties count as one half). Random ranking has an expected AUC of 0.5 and perfect separation gives 1.0. No
threshold is needed, and only the ranking matters, not calibration. The example submission is constant and therefore scores ~0.5.

The evaluation rows are split between a public leaderboard and private leaderboard set randomly.
Submission scoring computes the ROC AUC for both, but only the public leaderboard score will get updated during Phase 1 of the competition.
The final ranking will be determined by the private AUC, to be revealed only after the end of Phase 1.
ROC AUC is computed over all rows in the public/private set rows, and is not a per-user or per-event average.

**Each team is limited to 10 submissions per day!**

## Starter solution

A simple starter solution is provided in starter_solution.py.

To run it yourself, use the command
```
python3 starter_solution.py --train train.csv --evaluation evaluation.csv --out submission.csv
```

## Checklist

- [ ] Read the CSV files with a CSV parser and parse the history columns as JSON.
- [ ] Keep identifiers as text and keep each prediction attached to its `rowid`.
- [ ] Produce one prediction for every evaluation row.
- [ ] Check the header, the number of rows, duplicates, missing or unknown `rowid`s, and that all values are in [0, 1].
- [ ] Check your submission status before retrying an upload.
