import numpy as np
import pytest

N_FEATURES = 5
TEST_LEN = {"machine-1-1": 200, "machine-1-2": 201}  # machine-1-2: one extra test row


@pytest.fixture
def smd_root(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "ServerMachineDataset"
    for d in ["train", "test", "test_label", "interpretation_label"]:
        (root / d).mkdir(parents=True)
    for m, n_test in TEST_LEN.items():
        np.savetxt(root / "train" / f"{m}.txt", rng.random((200, N_FEATURES)), delimiter=",")
        np.savetxt(root / "test" / f"{m}.txt", rng.random((n_test, N_FEATURES)), delimiter=",")
        labels = np.zeros(n_test, dtype=int)
        labels[50:60] = 1
        np.savetxt(root / "test_label" / f"{m}.txt", labels, fmt="%d")
        (root / "interpretation_label" / f"{m}.txt").write_text("51-60:1,3\n")
    return root


@pytest.fixture
def hdfs_root(tmp_path):
    root = tmp_path / "HDFS_v1"
    (root / "preprocessed").mkdir(parents=True)
    lines = [
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_1 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_1.xml. blk_1",
        "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1 terminating",
        "081109 203520 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_2 src: /10.1.2.3:5000 dest: /10.1.2.4:50010",
        "081109 203520 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/other/job_77.xml. blk_2",
        "081109 203521 145 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_2 terminating",
        "081109 203522 146 INFO dfs.DataNode$DataXceiver: Receiving block blk_3 src: /10.1.1.1:1 dest: /10.1.1.2:2",
        "081109 203523 147 WARN dfs.DataNode$DataXceiver: writeBlock blk_3 received exception java.io.IOException: Connection reset",
        "this is not a valid log line",
    ]
    (root / "HDFS.log").write_text("\n".join(lines) + "\n")
    (root / "preprocessed" / "anomaly_label.csv").write_text(
        "BlockId,Label\nblk_1,Normal\nblk_2,Normal\nblk_3,Anomaly\n")
    (root / "preprocessed" / "HDFS_templates.csv").write_text(
        "EventId,EventTemplate\n"
        "E1,Receiving block <*> src: <*> dest: <*>\n"
        "E2,BLOCK* NameSystem.allocateBlock: <*> <*>\n"
        "E3,PacketResponder <*> for block <*> terminating\n"
        "E4,writeBlock <*> received exception <*>\n")
    return root