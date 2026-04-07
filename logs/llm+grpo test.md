# CASCADE CONTAINMENT — GRPO EVALUATION

Rollouts per task:  {'easy': 3, 'medium': 4, 'hard': 4}
  Learning:           Episodic memory + advantage gating

  Task: EASY | 3 rollouts
  ────────────────────────────────────────────

    Rollout 1/3 [base prompt]
      step  1: allocate → district 1 | reward: +0.3516
      step  2: allocate → district 1 | reward: +0.7000
      step  3: allocate → district 1 | reward: +0.6500
      step  4: allocate → district 1 | reward: +0.6000
      step  5: allocate → district 1 | reward: +0.0000
      step  6: allocate → district 1 | reward: +0.2000
      step  7: allocate → district 0 | reward: +0.3000
    → Grader: containment=1.000 hospital=0.999 efficiency=0.857 speed=0.300
    → Reward: +2.8016 | Score: 0.9083
    → Advantage: +0.0000 | ↑ Stored 7 steps

    Rollout 2/3 [memory: 7 entries]
      step  1: allocate → district 1 | reward: +0.0571
      step  2: allocate → district 1 | reward: +0.7000
      step  3: allocate → district 1 | reward: +0.3000
      step  4: allocate → district 0 | reward: +0.3000
      step  5: allocate → district 1 | reward: +0.3000
      step  6: allocate → district 0 | reward: +0.2000
      step  7: allocate → district 1 | reward: +0.3000
      step  8: allocate → district 0 | reward: +0.1000
      step  9: allocate → district 1 | reward: +0.0000
      step 10: allocate → district 0 | reward: +0.0000
    → Grader: containment=1.000 hospital=1.000 efficiency=0.700 speed=0.000
    → Reward: +2.2571 | Score: 0.8548
    → Advantage: -0.5445 | ↓ Suppressed

    Rollout 3/3 [memory: 7 entries]
      step  1: allocate → district 1 | reward: +0.2692
      step  2: allocate → district 1 | reward: +0.7000
      step  3: allocate → district 1 | reward: +0.6500
      step  4: allocate → district 1 | reward: +0.6000
      step  5: allocate → district 1 | reward: +0.0000
      step  6: allocate → district 0 | reward: +0.2000
      step  7: allocate → district 1 | reward: +0.4500
      step  8: allocate → district 1 | reward: +0.1000
      step  9: allocate → district 0 | reward: +0.0500
      step 10: allocate → district 1 | reward: +0.0000
    → Grader: containment=1.000 hospital=1.000 efficiency=1.000 speed=0.000
    → Reward: +3.0192 | Score: 0.8998
    → Advantage: +0.4899 | ↑ Stored 10 steps

  Rewards:    [2.8016, 2.2571, 3.0192]
  Mean:       +2.6926
  Advantages: [0.109, -0.4355, 0.3266]
  Best score: 0.9083

  ✓ EASY final score: 0.9083

  Task: MEDIUM | 4 rollouts
  ────────────────────────────────────────────

    Rollout 1/4 [base prompt]
      step  1: allocate → district 0 | reward: +0.3321
      step  2: allocate → district 2 | reward: -0.5170
      step  3: allocate → district 0 | reward: -0.6012
      step  4: allocate → district 2 | reward: -0.5170
      step  5: allocate → district 0 | reward: -1.4830
      step  6: allocate → district 3 | reward: -1.6858
      step  7: allocate → district 1 | reward: -2.0000
      step  8: allocate → district 2 | reward: -2.0000
      step  9: allocate → district 0 | reward: -2.0000
      step 10: allocate → district 3 | reward: -2.0000
      step 11: allocate → district 1 | reward: -2.0000
      step 12: allocate → district 0 | reward: -2.0000
      step 13: allocate → district 2 | reward: -2.0000
      step 14: allocate → district 3 | reward: -2.0000
      step 15: allocate → district 1 | reward: -2.0000
    → Grader: containment=0.154 hospital=0.853 efficiency=1.000 speed=0.000
    → Reward: -22.4719 | Score: 0.5802
    → Advantage: +0.0000 | ↑ Stored 1 steps

    Rollout 2/4 [memory: 1 entries]
      step  1: allocate → district 0 | reward: +0.4085
      step  2: allocate → district 2 | reward: -0.1591
      step  3: allocate → district 0 | reward: -0.5248
      step  4: allocate → district 2 | reward: -0.2248
      step  5: allocate → district 2 | reward: -0.5924
      step  6: allocate → district 0 | reward: -0.9312
      step  7: allocate → district 3 | reward: -1.1172
      step  8: allocate → district 2 | reward: -1.5236
      step  9: allocate → district 0 | reward: -2.0000
      step 10: allocate → district 2 | reward: -2.0000
      step 11: allocate → district 3 | reward: -2.0000
      step 12: allocate → district 2 | reward: -2.0000
      step 13: allocate → district 0 | reward: -2.0000
      step 14: allocate → district 3 | reward: -2.0000
      step 15: allocate → district 2 | reward: -2.0000
    → Grader: containment=0.269 hospital=0.940 efficiency=1.000 speed=0.000
    → Reward: -18.6646 | Score: 0.6539
    → Advantage: +3.8073 | ↑ Stored 3 steps

    Rollout 3/4 [memory: 4 entries]
      step  1: allocate → district 0 | reward: +0.6527
      step  2: allocate → district 2 | reward: -0.2589
      step  3: allocate → district 0 | reward: +0.4000
      step  4: allocate → district 2 | reward: -0.6922
      step  5: allocate → district 0 | reward: +0.0000
      step  6: allocate → district 3 | reward: -0.2806
      step  7: allocate → district 2 | reward: -1.1172
      step  8: allocate → district 0 | reward: -0.4250
      step  9: allocate → district 3 | reward: -1.3978
      step 10: allocate → district 0 | reward: -0.7056
      step 11: allocate → district 2 | reward: -2.0000
      step 12: allocate → district 3 | reward: -2.0000
      step 13: allocate → district 0 | reward: -2.0000
      step 14: allocate → district 3 | reward: -2.0000
      step 15: allocate → district 2 | reward: -2.0000
    → Grader: containment=0.423 hospital=0.976 efficiency=1.000 speed=0.000
    → Reward: -13.8246 | Score: 0.7161
    → Advantage: +6.7436 | ↑ Stored 5 steps

    Rollout 4/4 [memory: 9 entries]
      step  1: allocate → district 0 | reward: +0.4225
      step  2: allocate → district 2 | reward: +0.0281
      step  3: allocate → district 0 | reward: -0.5108
      step  4: allocate → district 2 | reward: +0.0000
      step  5: allocate → district 0 | reward: -0.5108
      step  6: allocate → district 2 | reward: -0.6728
      step  7: allocate → district 1 | reward: -1.3272
      step  8: allocate → district 3 | reward: -1.5888
      step  9: allocate → district 0 | reward: -2.0000
      step 10: allocate → district 2 | reward: -2.0000
      step 11: allocate → district 1 | reward: -2.0000
      step 12: allocate → district 3 | reward: -2.0000
      step 13: allocate → district 2 | reward: -2.0000
      step 14: allocate → district 0 | reward: -2.0000
      step 15: allocate → district 1 | reward: -2.0000
    → Grader: containment=0.288 hospital=0.930 efficiency=1.000 speed=0.000
    → Reward: -18.1598 | Score: 0.6552
    → Advantage: +0.1606 | ↑ Stored 3 steps

  Rewards:    [-22.4719, -18.6646, -13.8246, -18.1598]
  Mean:       -18.2802
  Advantages: [-4.1917, -0.3844, 4.4556, 0.1204]
  Best score: 0.7161

  ✓ MEDIUM final score: 0.7161

  Task: HARD | 4 rollouts
  ────────────────────────────────────────────

    Rollout 1/4 [base prompt]
      step  1: allocate → district 1 | reward: +1.4000
      step  2: allocate → district 2 | reward: +0.4333
      step  3: allocate → district 2 | reward: +0.4000
      step  4: allocate → district 2 | reward: +0.1167
      step  5: allocate → district 2 | reward: -0.9538
      step  6: allocate → district 0 | reward: -1.6567
      step  7: allocate → district 0 | reward: -1.4400
      step  8: allocate → district 0 | reward: -2.2308
      step  9: allocate → district 1 | reward: -2.2308
      step 10: allocate → district 3 | reward: -2.4808
      step 11: allocate → district 1 | reward: -3.0010
      step 12: allocate → district 1 | reward: -3.0010
      step 13: allocate → district 3 | reward: -3.0010
      step 14: allocate → district 3 | reward: -3.0010
      step 15: allocate → district 3 | reward: -3.0010
    → Grader: containment=0.308 hospital=0.906 efficiency=0.600 speed=0.000
    → Reward: -23.6479 | Score: 0.5901
    → Advantage: +0.0000 | ↑ Stored 4 steps

    Rollout 2/4 [memory: 4 entries]
      step  1: allocate → district 4 | reward: +0.9333
      step  2: allocate → district 2 | reward: +0.0000
      step  3: allocate → district 4 | reward: +0.4000
      step  4: allocate → district 4 | reward: -0.3362
      step  5: allocate → district 0 | reward: -0.7029
      step  6: allocate → district 1 | reward: -1.3884
      step  7: allocate → district 1 | reward: -1.3884
      step  8: allocate → district 0 | reward: -1.3884
      step  9: allocate → district 5 | reward: -1.8537
      step 10: allocate → district 2 | reward: -3.0000
      step 11: allocate → district 5 | reward: -3.0000
      step 12: allocate → district 5 | reward: -3.0000
      step 13: allocate → district 2 | reward: -3.0000
      step 14: allocate → district 5 | reward: -3.0000
      step 15: allocate → district 3 | reward: -2.7000
    → Grader: containment=0.346 hospital=0.890 efficiency=0.533 speed=0.000
    → Reward: -23.4247 | Score: 0.5843
    → Advantage: +0.2232 | ↑ Stored 3 steps

    Rollout 3/4 [memory: 7 entries]
      step  1: allocate → district 4 | reward: +0.9333
      step  2: allocate → district 4 | reward: +1.3000
      step  3: allocate → district 4 | reward: +0.4000
      step  4: allocate → district 4 | reward: -0.0878
      step  5: allocate → district 5 | reward: +0.3333
      step  6: allocate → district 5 | reward: +0.3000
      step  7: allocate → district 5 | reward: -0.7977
      step  8: allocate → district 5 | reward: -1.0396
      step  9: allocate → district 5 | reward: -1.9036
      step 10: allocate → district 2 | reward: -2.3788
      step 11: allocate → district 0 | reward: -2.5455
      step 12: allocate → district 0 | reward: -2.0703
      step 13: allocate → district 1 | reward: -2.0703
      step 14: allocate → district 4 | reward: -2.7000
      step 15: allocate → district 4 | reward: -3.0000
    → Grader: containment=0.513 hospital=0.971 efficiency=0.467 speed=0.000
    → Reward: -15.3270 | Score: 0.6608
    → Advantage: +8.2093 | ↑ Stored 6 steps

    Rollout 4/4 [memory: 13 entries]
      step  1: allocate → district 4 | reward: +0.9333
      step  2: allocate → district 4 | reward: +0.8667
      step  3: allocate → district 4 | reward: +0.8000
      step  4: allocate → district 4 | reward: -0.1832
      step  5: allocate → district 4 | reward: -0.2166
      step  6: allocate → district 5 | reward: -0.8145
      step  7: allocate → district 5 | reward: -0.2979
      step  8: allocate → district 5 | reward: -1.4760
      step  9: allocate → district 5 | reward: -1.4760
      step 10: allocate → district 5 | reward: -1.8331
      step 11: allocate → district 2 | reward: -2.4498
      step 12: allocate → district 2 | reward: -2.4498
      step 13: allocate → district 0 | reward: -2.4498
      step 14: allocate → district 1 | reward: -2.9997
      step 15: allocate → district 0 | reward: -2.9997
    → Grader: containment=0.462 hospital=0.959 efficiency=0.467 speed=0.000
    → Reward: -17.0461 | Score: 0.6398
    → Advantage: +3.7538 | ↑ Stored 6 steps

  Rewards:    [-23.6479, -23.4247, -15.327, -17.0461]
  Mean:       -19.8614
  Advantages: [-3.7865, -3.5633, 4.5344, 2.8153]
  Best score: 0.6608

  ✓ HARD final score: 0.6608

## FINAL SCORES

Easy:    0.9083
Medium:  0.7161
Hard:    0.6608
────────────────────────────────
Average: 0.7617
Time:    1186.2s
