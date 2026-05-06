#!/usr/bin/env python3
"""Extract one transformer FFN layer's weights from a Qwen safetensors
checkpoint and write them as raw float32 binaries (column-major) for the
test_ffn_layer_real_qwen.cu CUDA bench. Pure stdlib + numpy — no torch
required.

Usage:
    python scripts/demo/extract_qwen_layer.py \
        --model ~/models/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/<sha> \
        --layer 10 \
        --out /tmp/qwen_layer_weights
"""
import argparse, json, os, glob, struct
import numpy as np

DTYPE_MAP = {'F32': np.float32, 'F16': np.float16, 'F64': np.float64,
             'I64': np.int64, 'I32': np.int32, 'BF16': 'bf16'}
ITEMSIZE = {'F32':4, 'F16':2, 'BF16':2, 'F64':8, 'I64':8, 'I32':4}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='HF snapshot directory')
    ap.add_argument('--layer', type=int, default=10)
    ap.add_argument('--out', default='/tmp/qwen_layer_weights')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shards = sorted(glob.glob(f'{args.model}/model-*.safetensors'))
    keys = {f'model.layers.{args.layer}.mlp.{name}_proj.weight': name
            for name in ('gate', 'up', 'down')}
    for shard in shards:
        with open(shard, 'rb') as f:
            hsz = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(hsz).decode('utf-8'))
            data_off = 8 + hsz
            for k, val in header.items():
                if k in keys:
                    name = keys[k]
                    dtype = val['dtype']; shape = val['shape']
                    beg, end = val['data_offsets']
                    f.seek(data_off + beg)
                    raw = f.read(end - beg)
                    if dtype == 'BF16':
                        bf = np.frombuffer(raw, dtype=np.uint16)
                        arr = (bf.astype(np.uint32) << 16).view(np.float32).reshape(shape)
                    else:
                        arr = np.frombuffer(raw, dtype=DTYPE_MAP[dtype]).astype(np.float32).reshape(shape)
                    arr_T = np.ascontiguousarray(arr.T)
                    out_path = f'{args.out}/{name}_proj.f32'
                    arr_T.tofile(out_path)
                    print(f'{name}_proj: HF[{arr.shape}] BF16 -> col-major[{arr_T.shape}] '
                          f'fp32 -> {out_path} ({arr_T.nbytes/1e6:.1f} MB)')

if __name__ == '__main__':
    main()
