# Coevolution pilot: data audit

Matched specialist roles; one prespecified graph from each of the ER and WS families.
20 statement groups are split into 12 fitting, 4 validation, and 4 reserved assessment statements.
The inspected schema example `false_0` is assigned to fitting. Splits are balanced by the source true/false statement labels, which are not predictor inputs.

| Partition | Agent updates | Opinion changes | Persistence error | Neighbor-majority error |
| --- | ---: | ---: | ---: | ---: |
| train | 11520 | 1326 | 11.51% | 14.98% |
| validation | 3840 | 126 | 3.28% | 4.95% |

All native labels were checked against the argmax of recorded probability distributions. Graph edges were checked against every agent’s recorded inbox neighbors.
Assessment outcomes were not summarized or scored. Only development partitions contribute the table above.

Source: https://github.com/carlomarxdk/coevolve-sim
