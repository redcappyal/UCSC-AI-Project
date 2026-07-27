// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
// ios/Sources/Peer/BLETransport.swift
import CoreBluetooth
import Foundation

enum BLEChunker {
    static func chunks(_ frame: Data, maxWriteLength: Int) -> [Data] {
        let step = max(1, maxWriteLength)
        return stride(from: 0, to: frame.count, by: step).map {
            frame.subdata(in: frame.startIndex + $0 ..< frame.startIndex + min($0 + step, frame.count))
        }
    }
}

/// Initiator = CBCentralManager (scans, connects, writes control with
/// response / datagrams without response). Responder = CBPeripheralManager
/// (advertises, receives writes, sends via indicate/notify). All delegate
/// work on `queue`; callbacks fire on `queue`.
final class BLETransport: NSObject, PeerTransport {
    static let serviceUUID = CBUUID(string: "8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A11")
    static let controlUUID = CBUUID(string: "8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A12")
    static let datagramUUID = CBUUID(string: "8E4C6F2A-1B0D-4F3E-9A57-C3D2E1F00A13")

    let name = "ble"
    var onControl: ((Data) -> Void)?
    var onDatagram: ((Data) -> Void)?
    var onStateChange: ((TransportState) -> Void)?

    private let queue: DispatchQueue
    private var reassembler = FrameReassembler()

    // Central (initiator) state
    private var central: CBCentralManager?
    private var remotePeripheral: CBPeripheral?
    private var remoteControl: CBCharacteristic?
    private var remoteDatagram: CBCharacteristic?

    // Peripheral (responder) state
    private var peripheralManager: CBPeripheralManager?
    private var localControl: CBMutableCharacteristic?
    private var localDatagram: CBMutableCharacteristic?
    private var subscribedCentral: CBCentral?
    /// Peripheral role only: the central must subscribe to BOTH
    /// characteristics before we fire `.connected` — PeerSession sends hello
    /// via control-indicate the instant `.connected` fires, so signaling on
    /// datagram-subscribe alone can race the central's control subscription
    /// (still in flight) and silently drop the hello, wedging pairing.
    private var controlSubscribed = false
    private var datagramSubscribed = false
    /// Guards `.connected` firing more than once (each subscribe event is
    /// independent; only the one that completes the pair should signal).
    private var didEmitConnected = false
    /// Control chunks that hit a queue-full (`updateValue` returned false)
    /// indicate; drained in order from `peripheralManagerIsReady`.
    private var pendingIndications: [Data] = []
    /// Set once `tearDown()` runs; stays true for the life of the instance
    /// (a torn-down transport is dead — new sessions create new transports).
    /// Lets delegate callbacks distinguish a stale, already-handled
    /// completion (e.g. CoreBluetooth's async cancel finishing after we've
    /// already emitted our own `.disconnected`) from a live one.
    private var isTearingDown = false

    init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.ble")) {
        self.queue = queue
        super.init()
    }

    func startInitiator() {
        queue.async { [weak self] in
            guard let self else { return }
            self.onStateChange?(.searching)
            self.central = CBCentralManager(delegate: self, queue: self.queue)
        }
    }

    func startResponder() {
        queue.async { [weak self] in
            guard let self else { return }
            self.onStateChange?(.searching)
            self.peripheralManager = CBPeripheralManager(delegate: self, queue: self.queue)
        }
    }

    func sendControl(_ frame: Data) {
        queue.async { [weak self] in
            guard let self else { return }
            let encoded = FrameCodec.encode(frame)
            if let peripheral = self.remotePeripheral, let characteristic = self.remoteControl {
                let maxLength = peripheral.maximumWriteValueLength(for: .withResponse)
                for chunk in BLEChunker.chunks(encoded, maxWriteLength: maxLength) {
                    peripheral.writeValue(chunk, for: characteristic, type: .withResponse)
                }
            } else if let manager = self.peripheralManager, let characteristic = self.localControl {
                let maxLength = self.subscribedCentral?.maximumUpdateValueLength ?? 180
                self.enqueueIndications(BLEChunker.chunks(encoded, maxWriteLength: maxLength),
                                        manager: manager, characteristic: characteristic)
            }
        }
    }

    func sendDatagram(_ datagram: Data) {
        queue.async { [weak self] in
            guard let self else { return }
            if let peripheral = self.remotePeripheral, let characteristic = self.remoteDatagram {
                guard datagram.count <= peripheral.maximumWriteValueLength(for: .withoutResponse) else { return }
                peripheral.writeValue(datagram, for: characteristic, type: .withoutResponse)
            } else if let manager = self.peripheralManager, let characteristic = self.localDatagram {
                guard datagram.count <= (self.subscribedCentral?.maximumUpdateValueLength ?? 180) else { return }
                _ = manager.updateValue(datagram, for: characteristic, onSubscribedCentrals: nil)
            }
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.tearDown()
            self.onStateChange?(.disconnected("stopped"))
        }
    }

    /// Sends `chunks` via `updateValue` in order; on the first queue-full
    /// (false) return, stops and queues that chunk plus everything after it
    /// in `pendingIndications` for `peripheralManagerIsReady` to drain.
    /// Only used for the control channel — datagrams stay fire-and-forget.
    private func enqueueIndications(_ chunks: [Data], manager: CBPeripheralManager,
                                    characteristic: CBMutableCharacteristic) {
        for (index, chunk) in chunks.enumerated() {
            if !manager.updateValue(chunk, for: characteristic, onSubscribedCentrals: nil) {
                pendingIndications.append(contentsOf: chunks[index...])
                return
            }
        }
    }

    /// Full connection-state reset, shared by `stop()` and the corruption
    /// path in `ingestControlChunk`. Callers are responsible for emitting
    /// their own `onStateChange` event around this call.
    private func tearDown() {
        isTearingDown = true
        if let remotePeripheral { central?.cancelPeripheralConnection(remotePeripheral) }
        central?.stopScan()
        peripheralManager?.stopAdvertising()
        central = nil
        peripheralManager = nil
        remotePeripheral = nil
        remoteControl = nil
        remoteDatagram = nil
        subscribedCentral = nil
        localControl = nil
        localDatagram = nil
        reassembler = FrameReassembler()
        pendingIndications = []
        controlSubscribed = false
        datagramSubscribed = false
        didEmitConnected = false
    }

    private func ingestControlChunk(_ chunk: Data) {
        guard let frames = try? reassembler.ingest(chunk) else {
            onStateChange?(.disconnected("control stream corrupted"))
            tearDown()
            return
        }
        for frame in frames { onControl?(frame) }
    }
}

extension BLETransport: CBCentralManagerDelegate, CBPeripheralDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            central.scanForPeripherals(withServices: [Self.serviceUUID])
        case .poweredOff, .unauthorized, .unsupported:
            onStateChange?(.disconnected("bluetooth unavailable: \(central.state)"))
            tearDown()
        case .resetting, .unknown:
            break
        @unknown default:
            break
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        remotePeripheral = peripheral
        peripheral.delegate = self
        central.stopScan()
        central.connect(peripheral)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.discoverServices([Self.serviceUUID])
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral,
                        error: Error?) {
        guard !isTearingDown else { return }
        onStateChange?(.disconnected(error?.localizedDescription ?? "connect failed"))
        tearDown()
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        guard !isTearingDown else { return }
        onStateChange?(.disconnected(error?.localizedDescription ?? "peer disconnected"))
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let service = peripheral.services?.first(where: { $0.uuid == Self.serviceUUID }) else { return }
        peripheral.discoverCharacteristics([Self.controlUUID, Self.datagramUUID], for: service)
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        for characteristic in service.characteristics ?? [] {
            if characteristic.uuid == Self.controlUUID {
                remoteControl = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
            } else if characteristic.uuid == Self.datagramUUID {
                remoteDatagram = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
            }
        }
        if remoteControl != nil, remoteDatagram != nil { onStateChange?(.connected) }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic,
                    error: Error?) {
        guard !isTearingDown, let error else { return }
        onStateChange?(.disconnected("subscribe failed: \(error.localizedDescription)"))
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let value = characteristic.value else { return }
        if characteristic.uuid == Self.controlUUID { ingestControlChunk(value) }
        else if characteristic.uuid == Self.datagramUUID { onDatagram?(value) }
    }
}

extension BLETransport: CBPeripheralManagerDelegate {
    func peripheralManagerDidUpdateState(_ manager: CBPeripheralManager) {
        switch manager.state {
        case .poweredOn:
            let control = CBMutableCharacteristic(type: Self.controlUUID,
                                                  properties: [.write, .indicate],
                                                  value: nil, permissions: [.writeable])
            let datagram = CBMutableCharacteristic(type: Self.datagramUUID,
                                                   properties: [.writeWithoutResponse, .notify],
                                                   value: nil, permissions: [.writeable])
            localControl = control; localDatagram = datagram
            let service = CBMutableService(type: Self.serviceUUID, primary: true)
            service.characteristics = [control, datagram]
            manager.add(service)
        case .poweredOff, .unauthorized, .unsupported:
            onStateChange?(.disconnected("bluetooth unavailable: \(manager.state)"))
            tearDown()
        case .resetting, .unknown:
            break
        @unknown default:
            break
        }
    }

    func peripheralManager(_ manager: CBPeripheralManager, didAdd service: CBService, error: Error?) {
        if let error {
            onStateChange?(.disconnected("service add failed: \(error.localizedDescription)"))
            tearDown()
            return
        }
        guard manager.state == .poweredOn else { return }
        manager.startAdvertising([CBAdvertisementDataServiceUUIDsKey: [Self.serviceUUID]])
    }

    func peripheralManager(_ manager: CBPeripheralManager, central: CBCentral,
                           didSubscribeTo characteristic: CBCharacteristic) {
        subscribedCentral = central
        if characteristic.uuid == Self.controlUUID { controlSubscribed = true }
        else if characteristic.uuid == Self.datagramUUID { datagramSubscribed = true }
        // Fire exactly once, and only once BOTH characteristics are
        // subscribed — PeerSession sends hello via control-indicate right
        // on `.connected`, and that hello is silently dropped if the
        // central hasn't finished subscribing to control yet.
        if controlSubscribed, datagramSubscribed, !didEmitConnected {
            didEmitConnected = true
            onStateChange?(.connected)
        }
    }

    func peripheralManager(_ manager: CBPeripheralManager,
                           didReceiveWrite requests: [CBATTRequest]) {
        for request in requests {
            guard let value = request.value else { continue }
            if request.characteristic.uuid == Self.controlUUID {
                ingestControlChunk(value)
                manager.respond(to: request, withResult: .success)
            } else if request.characteristic.uuid == Self.datagramUUID {
                onDatagram?(value)
            }
        }
    }

    func peripheralManagerIsReady(toUpdateSubscribers manager: CBPeripheralManager) {
        guard let characteristic = localControl else { return }
        let queued = pendingIndications
        pendingIndications = []
        enqueueIndications(queued, manager: manager, characteristic: characteristic)
    }
}
