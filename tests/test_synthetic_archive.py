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


def _write_archive(tmp_path, metadata):
    p=tmp_path/"case.npz"
    np.savez(p,features=np.zeros((2,1,3,4,5),np.float32),
             labels=np.array([0,1],dtype=np.int64),
             subject_ids=np.array(["s01","s02"]),
             trial_ids=np.array([0,1],dtype=np.int64),
             adjacency=np.eye(3,dtype=np.float32),
             metadata_json=np.asarray(json.dumps(metadata)))
    return p


def _real_metadata():
    return {
        "schema_version":"1.1","provenance":"real","dataset_name":"DEAP",
        "dataset_version":"exact downloaded release to be recorded",
        "source_release":"provider-preprocessed MATLAB variant",
        "feature_method":"time_frequency_features",
        "channel_names":["F3","F4","C3"],"sampling_rate_hz":128,
        "preprocessing":{"filter_cutoffs_hz":[4,30],"filter_order_status":"provisional"},
        "random_seed":42,
        "feature_generation":{"method":"DWT","parameters":{"wavelet":"db4","level":None},
                              "parameter_status":"partially_confirmed"},
        "frequency_bands_hz":[{"name":"theta","low_hz":4.0,"high_hz":8.0}],
        "label_mapping":{"class_names":["negative","positive"],
                         "rule":"exact threshold and equality rule to be recorded"},
        "graph_construction":{"connectivity":"PLV","sparsification":"top-k",
                              "normalization":"symmetric"},
        "evaluation_protocol":{"strategy":"LOSO","split_definition":"held-out subject per fold",
                               "validation_policy":"training subjects only"},
        "software_environment":{"python":"3.x","packages":{"numpy":"version","torch":"version"}},
        "data_access":{"access_status":"authorized",
                       "usage_note":"Accessed under dataset terms; recordings not redistributed."}
    }


def test_real_archive_requires_traceability_fields(tmp_path):
    metadata=_real_metadata()
    metadata.pop("graph_construction")
    with pytest.raises(ValueError,match="Real archive metadata is missing"):
        load_feature_archive(_write_archive(tmp_path,metadata))


def test_real_archive_validates_nested_feature_metadata(tmp_path):
    metadata=_real_metadata()
    metadata["feature_generation"]["parameters"]="not-an-object"
    with pytest.raises(ValueError,match="feature_generation.parameters"):
        load_feature_archive(_write_archive(tmp_path,metadata))


def test_real_archive_accepts_explicit_unresolved_parameters(tmp_path):
    loaded=load_feature_archive(_write_archive(tmp_path,_real_metadata()))
    assert loaded["metadata"]["provenance"]=="real"
    assert loaded["metadata"]["feature_generation"]["parameters"]["level"] is None


def test_real_archive_rejects_old_schema(tmp_path):
    metadata=_real_metadata()
    metadata["schema_version"]="1.0"
    with pytest.raises(ValueError,match="Real archives require"):
        load_feature_archive(_write_archive(tmp_path,metadata))
