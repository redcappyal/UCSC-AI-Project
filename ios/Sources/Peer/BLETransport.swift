// ios/Sources/Peer/BLETransport.swift
import CoreBluetooth
import Foundation

enum BLEChunker {
    static func chunks(_ frame: Data, maxWriteLength: Int) -> [Data] {
        stride(from: 0, to: frame.count, by: maxWriteLength).map {
            frame.subdata(in: frame.startIndex + $0 ..< frame.startIndex + min($0 + maxWriteLength, frame.count))
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
    private let reassembler = FrameReassembler()

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

    init(queue: DispatchQueue = DispatchQueue(label: "slc.peer.ble")) {
        self.queue = queue
        super.init()
    }

    func startInitiator() {
        onStateChange?(.searching)
        central = CBCentralManager(delegate: self, queue: queue)
    }

    func startResponder() {
        onStateChange?(.searching)
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
            for chunk in BLEChunker.chunks(encoded, maxWriteLength: maxLength) {
                _ = manager.updateValue(chunk, for: characteristic, onSubscribedCentrals: nil)
            }
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
        if let remotePeripheral { central?.cancelPeripheralConnection(remotePeripheral) }
        central?.stopScan()
        peripheralManager?.stopAdvertising()
        central = nil; peripheralManager = nil
        onStateChange?(.disconnected("stopped"))
    }

    private func ingestControlChunk(_ chunk: Data) {
        guard let frames = try? reassembler.ingest(chunk) else {
            onStateChange?(.disconnected("control stream corrupted"))
            return
        }
        for frame in frames { onControl?(frame) }
    }
}

extension BLETransport: CBCentralManagerDelegate, CBPeripheralDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        guard central.state == .poweredOn else { return }
        central.scanForPeripherals(withServices: [Self.serviceUUID])
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

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let value = characteristic.value else { return }
        if characteristic.uuid == Self.controlUUID { ingestControlChunk(value) }
        else if characteristic.uuid == Self.datagramUUID { onDatagram?(value) }
    }
}

extension BLETransport: CBPeripheralManagerDelegate {
    func peripheralManagerDidUpdateState(_ manager: CBPeripheralManager) {
        guard manager.state == .poweredOn else { return }
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
}
