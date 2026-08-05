"""独立读取 SARS-Inter V2 外部补全 query 的最小契约。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
from collections.abc import Mapping, Sequence

import numpy as np


QUERY_FORMAT = 'sars_inter_external_completion_query'
QUERY_VERSION = 2
QUERY_KEYS = {
    'format', 'version', 'dataset_profile', 'source_split',
    'completion_scope', 'selection_manifest_hash', 'sample_order',
    'frame_count', 'joint_count', 'missing_mask_semantics',
    'coordinate_contract', 'temporal_metadata', 'masked_keypoint',
    'missing_mask', 'metadata', 'query_hash',
}
PRIVATE_EVALUATION_KEYS = {
    'label', 'labels', 'gt', 'gt_label', 'ground_truth',
    'clean_keypoint', 'clean_keypoints', 'clean_skeleton',
    'clean_reference_identifiers', 'recognition_scores', 'after_scores',
}


def _long_path(path):
    value = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        return '\\\\?\\' + value
    return value


def _array_descriptor(value):
    array = np.ascontiguousarray(np.asarray(value))
    return {
        'dtype': str(array.dtype),
        'shape': list(array.shape),
        'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
    }


def _hashable(value):
    if isinstance(value, np.ndarray):
        return {'__ndarray__': _array_descriptor(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('query mappings require string keys')
        return {
            key: _hashable(child)
            for key, child in sorted(value.items(), key=lambda item: item[0])
        }
    if isinstance(value, (list, tuple)):
        return [_hashable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('query floating metadata must be finite')
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError('unsupported value in query hash: {}'.format(type(value)))


def _query_hash(package):
    payload = {
        key: value for key, value in package.items() if key != 'query_hash'
    }
    encoded = json.dumps(
        _hashable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _reject_private_fields(value, location='metadata'):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    'query metadata keys must be strings at {}'.format(location)
                )
            if str(key).strip().lower() in PRIVATE_EVALUATION_KEYS:
                raise ValueError(
                    'private evaluation field is not allowed in query {}: {}'.format(
                        location, key
                    )
                )
            _reject_private_fields(child, '{}.{}'.format(location, key))
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_private_fields(child, '{}[{}]'.format(location, index))
        return
    if isinstance(value, (np.ndarray, np.generic)):
        raise TypeError(
            'query {} must contain JSON-like metadata, not numpy values'.format(
                location
            )
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('query metadata floating values must be finite')
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(
            'unsupported query metadata value at {}: {}'.format(
                location, type(value)
            )
        )


def validate_completion_query(package):
    if not isinstance(package, dict):
        raise TypeError('completion query must contain a dict')
    value = copy.deepcopy(package)
    if value.get('format') != QUERY_FORMAT or int(value.get('version', -1)) != QUERY_VERSION:
        raise ValueError('unsupported completion query format/version')
    missing_keys = QUERY_KEYS - set(value)
    if missing_keys:
        raise ValueError(
            'completion query is missing required fields: {}'.format(
                sorted(missing_keys)
            )
        )
    unsupported = set(value) - QUERY_KEYS
    if unsupported:
        raise ValueError(
            'completion query contains unsupported fields: {}'.format(
                sorted(unsupported)
            )
        )
    order = [str(sample_id) for sample_id in value.get('sample_order') or []]
    if not order or len(order) != len(set(order)):
        raise ValueError('sample_order must be non-empty and unique')
    if value.get('source_split') not in {'train', 'valid', 'test'}:
        raise ValueError('source_split must be train, valid, or test')
    scope = value.get('completion_scope')
    if scope not in {'all_split', 'trigger_selected'}:
        raise ValueError('unsupported completion_scope')
    if scope == 'trigger_selected' and not value.get('selection_manifest_hash'):
        raise ValueError('trigger_selected requires selection_manifest_hash')
    if int(value.get('frame_count', 0)) != 64 or int(value.get('joint_count', 0)) != 17:
        raise ValueError('SiC adapter requires H36M17/F64 query data')
    if value.get('missing_mask_semantics') != 'True=missing':
        raise ValueError('missing_mask_semantics must be True=missing')
    masked = np.asarray(value.get('masked_keypoint'))
    missing = np.asarray(value.get('missing_mask'))
    expected_keypoint = (len(order), 64, 17, 3)
    expected_mask = (len(order), 64, 17)
    if masked.shape != expected_keypoint:
        raise ValueError(
            'masked_keypoint shape must be {}; got {}'.format(
                expected_keypoint, masked.shape
            )
        )
    if missing.shape != expected_mask:
        raise ValueError(
            'missing_mask shape must be {}; got {}'.format(
                expected_mask, missing.shape
            )
        )
    if masked.dtype != np.float32:
        raise ValueError('masked_keypoint dtype must be float32')
    if missing.dtype != np.bool_:
        raise ValueError('missing_mask dtype must be bool')
    if not np.isfinite(masked).all():
        raise ValueError('masked_keypoint must contain finite values')
    for name in ('coordinate_contract', 'temporal_metadata', 'metadata'):
        _reject_private_fields(value.get(name) or {}, location=name)
    if value.get('query_hash') != _query_hash(value):
        raise ValueError('completion query query_hash mismatch')
    value['sample_order'] = order
    value['masked_keypoint'] = np.array(masked, copy=True)
    value['missing_mask'] = np.array(missing, copy=True)
    return value


def load_completion_query(path):
    with open(_long_path(path), 'rb') as stream:
        payload = pickle.load(stream)
    return validate_completion_query(payload)


__all__ = ['load_completion_query', 'validate_completion_query']
