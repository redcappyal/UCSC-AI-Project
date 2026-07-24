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
    /// Control chunks that hit a queue-full (`updateValue` returned false)
    /// indicate; drained in order from `peripheralManagerIsReady`.
    private var pendingIndications: [Data] = []

    init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.ble")) {
        self.queue = queue
        super.init()
    }

    func startInitiator() {
        queue.async { [weak self] in self?.onStateChange?(.searching) }
        central = CBCentralManager(delegate: self, queue: queue)
    }

    func startResponder() {
        queue.async { [weak self] in self?.onStateChange?(.searching) }
        peripheralManager = CBPeripheralManager(delegate: self, queue: queue)
    }

    func sendControl(_ frame: Data) {
        let encoded = FrameCodec.encode(frame)
        if let peripheral = remotePeripheral, let characteristic = remoteControl {
            let maxLength = peripheral.maximumWriteValueLength(for: .withResponse)
            for chunk in BLEChunker.chunks(encoded, maxWriteLength: maxLength) {
                peripheral.writeValue(chunk, for: characteristic, type: .withResponse)
            }
        } else if let manager = peripheralManager, let characteristic = localControl {
            let maxLength = subscribedCentral?.maximumUpdateValueLength ?? 180
            enqueueIndications(BLEChunker.chunks(encoded, maxWriteLength: maxLength),
                               manager: manager, characteristic: characteristic)
        }
    }

    func sendDatagram(_ datagram: Data) {
        if let peripheral = remotePeripheral, let characteristic = remoteDatagram {
            guard datagram.count <= peripheral.maximumWriteValueLength(for: .withoutResponse) else { return }
            peripheral.writeValue(datagram, for: characteristic, type: .withoutResponse)
        } else if let manager = peripheralManager, let characteristic = localDatagram {
            guard datagram.count <= (subscribedCentral?.maximumUpdateValueLength ?? 180) else { return }
            _ = manager.updateValue(datagram, for: characteristic, onSubscribedCentrals: nil)
        }
    }

    func stop() {
        tearDown()
        onStateChange?(.disconnected("stopped"))
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
        onStateChange?(.disconnected(error?.localizedDescription ?? "connect failed"))
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
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
        guard let error else { return }
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
        case .resetting, .unknown:
            break
        @unknown default:
            break
        }
    }

    func peripheralManager(_ manager: CBPeripheralManager, didAdd service: CBService, error: Error?) {
        if let error {
            onStateChange?(.disconnected("service add failed: \(error.localizedDescription)"))
            return
        }
        manager.startAdvertising([CBAdvertisementDataServiceUUIDsKey: [Self.serviceUUID]])
    }

    func peripheralManager(_ manager: CBPeripheralManager, central: CBCentral,
                           didSubscribeTo characteristic: CBCharacteristic) {
        subscribedCentral = central
        if characteristic.uuid == Self.datagramUUID { onStateChange?(.connected) }
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
