implementing ks;

public interface IHash<T>
{
    public uint64_t operator()(T x);
}

public interface IKeyEqual<T>
{
    public bool operator()(T x, T y);
}

public struct ReadonlyHashTable<Key, Value, Hash, KeyEqual>
    where Hash : IHash<Key>
    where KeyEqual : IKeyEqual<Key>
{
    public struct TableEntry
    {
        public bool has_value() { return set; }

        public Key key;
        public Value value;
        public bool set;
    }

    public __subscript(Key key)->Optional<Value>
    {
        get
        {
            uint64_t offset = find_offset(key);
            if (!table[offset].has_value()) {
                return none;
            }
            return Optional<Value>(table[offset].value);
        }
    }

    public bool try_get(Key key, out Value value)
    {
        uint64_t offset = find_offset(key);
        if (!table[offset].has_value()) {
            unused(value);
            return false;
        }
        value = table[offset].value;
        return true;
    }

    public uint64_t find_offset(Key key)
    {
        uint64_t base_offset = Hash()(key) & (capacity() - 1);
        for (int n_probes = 0;; ++n_probes) {
            // Find offset for _key_ using quadratic probing
            uint64_t offset = (base_offset + n_probes / 2 + n_probes * n_probes / 2) & (capacity() - 1);
            if (table[offset].has_value() == false || KeyEqual()(key, table[offset].key))
                return offset;
        }
    }

    public uint64_t capacity()
    {
        uint size, stride;
        table.GetDimensions(size, stride);
        unused(stride);
        return (uint64_t)size;
    }

    // TODO: double checkout alignment/padding, etc.
    public StructuredBuffer<TableEntry, ScalarDataLayout> table;
}
