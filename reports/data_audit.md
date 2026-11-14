# Released-data audit

Source revision: `62bf0e7ddb94e6be936bed6a35aefea94e8d54a9`.

1720 episodes: 1600 main and 120 supplemental low-rank episodes. 32 agents, 8 updates.

Question split: 16 train / 4 validation / 20 locked test; seed 1729.

| Question split | Graph role | Family | Episodes | Valid agent transitions |
| --- | --- | --- | ---: | ---: |
| test | fit_graph | random | 320 | 81920 |
| test | heldout_family | lowrank | 120 | 30720 |
| test | heldout_family | square | 80 | 20480 |
| test | heldout_family | triangular | 80 | 20480 |
| test | heldout_random | random | 320 | 81920 |
| train | fit_graph | random | 512 | 131072 |
| train | heldout_family | square | 64 | 16384 |
| train | heldout_family | triangular | 64 | 16384 |
| validation | fit_graph | random | 128 | 32768 |
| validation | heldout_family | square | 16 | 4096 |
| validation | heldout_family | triangular | 16 | 4096 |

Raw vote parse failures: 0; invalid majority states: 0.
Missing raw episodes: 0; invalid message rounds: 0.

Source J[sender,receiver] transposed to J[receiver,sender]; all observed source matrices symmetric.

All models exclude a snapshot if any agent has an invalid opinion in the current/target or available 2-snapshot history. Invalid votes excluded from counts; valid count retained.

At cutoff t use messages_history[t-1], empty at t=0; messages_history[t] excluded.

GPT-4o-mini has recipient-specific sampled messages; edge records retained.

Author question PCA artifact; fitting provenance not independently verified. Not claimed inductive.

Exact normalized duplicate questions: none. Semantic paraphrase groups are not established by this audit.

The low-rank subset covers ten official test questions, not all twenty. Test outcomes remain reserved for frozen model selection.
