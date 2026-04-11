# Sending Less, Learning Just as Much

---

## The Problem

Every training step, Machine 1 needs to send feedback to Machine 0.
Sending **everything**, every time, creates a **lot** of network traffic.

---

## The Idea

Not all feedback is equally important.

```
Each training step produces a grid of feedback values:

  [ 0.02  -0.71   0.04   0.88   0.01 ]
  [ 0.43  -0.03   0.90  -0.12   0.06 ]
  [ 0.08   0.61  -0.05   0.03   0.02 ]
         ...and thousands more...

          Rank by absolute size
                   |
        +----------+-----------+
        |                      |
   Top 10% (big)          Other 90% (small)
  [ 0.90, 0.88, -0.71 ]   [ 0.43, 0.08, 0.06 ... ]
        |                      |
   Send immediately         Add to accumulator
        |
        v                 Step 1:  [ 0.43   0.08   0.06 ]
  Machine 0                Step 2:  [ 0.61   0.19   0.14 ]  <- piling up
                           Step 3:  [ 0.74   0.31   0.20 ]
                           Step 4:  [ 0.93   0.44   0.28 ]
                                      ^
                                 now big enough!
                                      |
                                 Send + clear      -> Machine 0
```

---

## In Plain Terms

| What | What we do |
|---|---|
| Strong updates | Send immediately — they matter now |
| Weak updates | Save them up — send once they've built up |

> Think of it like holding small errands until a trip is worth making.

---

## Result

| | Data sent | Performance |
|---|---|---|
| Send everything | 133 GB | avg return 399 |
| Send selectively | **88 GB** ↓34% | avg return 305 *(76%)* |

---

## Talking Points

- Most of the feedback each step is small and redundant — we skip it for now
- The small signals aren't thrown away — they accumulate until meaningful
- 34% less data on the wire, policy still learns to drive the car
- The car still drives — just trained on a tighter communication budget
