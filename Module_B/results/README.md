# Module B results

> 历史实验：本目录采用内部开发集选模，不属于当前主协议。当前主结果请看 [results_ap](../results_ap/README.md)，原报告和权重保留供追溯。


COCO val2017, fixed detection boxes, single view.
All four methods use current-runtime predictions and the same scoring protocol.

| Method | AP |
| --- | ---: |
| P0 | 73.6223 |
| Fixed alpha=0.25 | 73.6652 |
| Top-1 alpha=1 | 73.5402 |
| B | 73.6263 |

B improves on Top-1, but does not beat fixed alpha=0.25.
The difference from P0 is small; no statistical significance claim is made.

The reports retain their original filenames, paths, fingerprints and source hashes.
Historical source code is in ../history.zip.
Large feature caches, gain shards, raw predictions and the base CIGPose checkpoint remain outside Git.
