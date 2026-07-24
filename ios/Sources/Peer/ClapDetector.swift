import Accelerate
import CoreMedia
import Foundation

/// Onset detector for the pairing clap. Pure core (`process(samples:...)`)
/// is unit-tested; the CMSampleBuffer adapter is exercised on hardware.
final class ClapDetector {
    var onClap: ((Double) -> Void)?

    private var noiseFloor: Float = 0.01
    private var lastOnset: Double = -.infinity
    private let windowSeconds = 0.005
    private let refractorySeconds = 0.5
    private let floorAlpha: Float = 0.05

    func process(samples: [Float], startTime: Double, sampleRate: Double) -> Double? {
        let window = max(1, Int(windowSeconds * sampleRate))
        var index = 0
        while index < samples.count {
            let upper = min(index + window, samples.count)
            let slice = Array(samples[index..<upper])
            var rms: Float = 0
            vDSP_rmsqv(slice, 1, &rms, vDSP_Length(slice.count))
            let windowStart = startTime + Double(index) / sampleRate
            let triggers = rms > max(8 * noiseFloor, 0.05)
            if triggers, windowStart - lastOnset > refractorySeconds {
                let gate = 4 * noiseFloor
                let offsetInWindow = slice.firstIndex { abs($0) > gate } ?? 0
                let onset = windowStart + Double(offsetInWindow) / sampleRate
                lastOnset = onset
                return onset
            }
            if !triggers {
                noiseFloor += floorAlpha * (rms - noiseFloor)
            }
            index = upper
        }
        return nil
    }

    func process(sampleBuffer: CMSampleBuffer) -> Double? {
        guard let format = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee,
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return nil }
        var length = 0
        var pointer: UnsafeMutablePointer<Int8>?
        guard CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
                                          totalLengthOut: &length, dataPointerOut: &pointer) == noErr,
              let raw = pointer else { return nil }

        let channels = max(1, Int(asbd.mChannelsPerFrame))
        var mono: [Float]
        if asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0 {
            let floats = UnsafeRawPointer(raw).assumingMemoryBound(to: Float.self)
            let frameCount = length / MemoryLayout<Float>.size / channels
            mono = (0..<frameCount).map { floats[$0 * channels] }
        } else {
            let ints = UnsafeRawPointer(raw).assumingMemoryBound(to: Int16.self)
            let frameCount = length / MemoryLayout<Int16>.size / channels
            mono = (0..<frameCount).map { Float(ints[$0 * channels]) / Float(Int16.max) }
        }
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        let onset = process(samples: mono, startTime: pts, sampleRate: asbd.mSampleRate)
        if let onset { onClap?(onset) }
        return onset
    }
}
