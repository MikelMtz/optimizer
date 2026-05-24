import sys, os, pickle, zipfile, torch

folder = 'kink_lenet_sweep/width/guess_uniform_w16/models/'
files = sorted([os.path.join(folder, f) for f in os.listdir(folder)])
if not files:
    print("No files found")
    sys.exit(1)

path = files[0]
print(f'Loading {path}')

def _load_via_python_zipfile(path):
    with zipfile.ZipFile(path, 'r') as zf:
        namelist = zf.namelist()
        pkl_path = [n for n in namelist if n.endswith('data.pkl')][0]
        prefix = pkl_path.split('data.pkl')[0]
        print(f'  Using prefix: {prefix}')

        raw_storages = {}
        data_prefix = prefix + 'data/'
        for name in namelist:
            if name.startswith(data_prefix) and not name.endswith('/'):
                key = name[len(data_prefix):]
                raw_storages[key] = zf.read(name)
        print(f'  Loaded {len(raw_storages)} storages')

        class _Unpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                if pid[0] == 'storage':
                    _, storage_cls, key, location, numel = pid
                    if key not in raw_storages:
                        return None
                    storage = torch.UntypedStorage.from_buffer(
                        raw_storages[key], byte_order=sys.byteorder
                    )
                    
                    dtype = torch.float32
                    if 'FloatStorage' in str(storage_cls): dtype = torch.float32
                    elif 'DoubleStorage' in str(storage_cls): dtype = torch.float64
                    elif 'LongStorage' in str(storage_cls): dtype = torch.int64
                    elif 'IntStorage' in str(storage_cls): dtype = torch.int32
                    elif 'ByteStorage' in str(storage_cls): dtype = torch.uint8
                    elif 'BoolStorage' in str(storage_cls): dtype = torch.bool
                    
                    return torch.storage.TypedStorage(
                        wrap_storage=storage, dtype=dtype, _internal=True
                    )
                return None

        with zf.open(pkl_path) as pkl:
            return _Unpickler(pkl).load()

ckpt = _load_via_python_zipfile(path)
print('Keys:', list(ckpt.keys()))
if 'good_models_state_dict' in ckpt:
    sd = ckpt['good_models_state_dict']
    first_key = next(iter(sd))
    print(f'First state_dict key: {first_key}, shape: {sd[first_key].shape}')
print('SUCCESS')
