import json
import numpy as np
import pytest
from data.feature_archive import load_feature_archive
from scripts.generate_synthetic_archive import generate_synthetic_archive


def test_synthetic_archive_is_repeatable_and_loadable(tmp_path):
    a=load_feature_archive(generate_synthetic_archive(tmp_path/"a.npz",seed=77))
    b=load_feature_archive(generate_synthetic_archive(tmp_path/"b.npz",seed=77))
    assert np.array_equal(a["features"],b["features"])
    assert np.array_equal(a["labels"],b["labels"])
    assert a["metadata"]["provenance"]=="synthetic"
    assert "not recorded EEG" in a["metadata"]["synthetic_data_note"]


def test_archive_requires_provenance_metadata(tmp_path):
    p=tmp_path/"missing.npz"
    np.savez(p,features=np.zeros((2,1,3,4,5),np.float32),
             labels=np.array([0,1]),subject_ids=np.array([0,1]),
             trial_ids=np.array([0,0]),adjacency=np.eye(3,dtype=np.float32))
    with pytest.raises(KeyError,match="metadata_json"):
        load_feature_archive(p)


def test_archive_rejects_channel_metadata_mismatch(tmp_path):
    p=generate_synthetic_archive(tmp_path/"good.npz",electrodes=5)
    with np.load(p,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    m=json.loads(str(data["metadata_json"].item()))
    m["channel_names"]=["one"]
    data["metadata_json"]=np.asarray(json.dumps(m))
    bad=tmp_path/"bad.npz"
    np.savez(bad,**data)
    with pytest.raises(ValueError,match="channel_names"):
        load_feature_archive(bad)
