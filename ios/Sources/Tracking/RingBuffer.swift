/// Fixed-capacity FIFO over an array. Oldest-first snapshot via `elements`.
struct RingBuffer<Element> {
    private var storage: [Element] = []
    private var next = 0
    let capacity: Int

    init(capacity: Int) {
        self.capacity = max(1, capacity)
        storage.reserveCapacity(self.capacity)
    }

    var count: Int { storage.count }

    mutating func append(_ element: Element) {
        if storage.count < capacity {
            storage.append(element)
        } else {
            storage[next] = element
        }
        next = (next + 1) % capacity
    }

    var elements: [Element] {
        guard storage.count == capacity else { return storage }
        return Array(storage[next...]) + Array(storage[..<next])
    }
}
