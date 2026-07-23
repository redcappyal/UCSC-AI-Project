import Foundation

struct UploadResponse: Decodable, Equatable {
    let ok: Bool
    let videoID: String
    let fps: Double?
    let frameCount: Int?
    let duration: Double?

    enum CodingKeys: String, CodingKey {
        case ok, fps, duration
        case videoID = "video_id"
        case frameCount = "frame_count"
    }
}

struct Hit: Decodable, Equatable, Identifiable {
    var id: Int { frame }
    let frame: Int
    let timestampSeconds: Double
    let call: String            // IN | OUT | UNKNOWN | AUDIO
    let marginPx: Double?
    let eventType: String?
    let hasTargetZone: Bool
    let hasWallDiagram: Bool

    enum CodingKeys: String, CodingKey {
        case frame, call
        case timestampSeconds = "timestamp_seconds"
        case marginPx = "margin_px"
        case eventType = "event_type"
        case targetZone = "target_zone"
        case wallDiagram = "wall_diagram"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        frame = try container.decode(Int.self, forKey: .frame)
        timestampSeconds = try container.decode(Double.self, forKey: .timestampSeconds)
        call = try container.decode(String.self, forKey: .call)
        marginPx = try container.decodeIfPresent(Double.self, forKey: .marginPx)
        eventType = try container.decodeIfPresent(String.self, forKey: .eventType)
        // Presence-only: the shapes are the server's business (opaque here).
        hasTargetZone = container.contains(.targetZone)
            && !((try? container.decodeNil(forKey: .targetZone)) ?? true)
        hasWallDiagram = container.contains(.wallDiagram)
            && !((try? container.decodeNil(forKey: .wallDiagram)) ?? true)
    }
}

extension Array where Element == Hit {
    /// Mirror of app.py front_wall_hits_from_payload.
    var frontWall: [Hit] {
        filter {
            $0.hasTargetZone && $0.hasWallDiagram
                && ($0.eventType == nil || $0.eventType == "wall" || $0.eventType == "unknown")
        }
    }
}

struct JobStatus: Decodable, Equatable {
    let ok: Bool
    let status: String          // queued | running | complete | failed
    let runID: String?
    let stage: String?
    let progress: Double?
    let processedFrames: Int?
    let totalFrames: Int?
    let message: String?
    let error: String?
    let hits: [Hit]?

    enum CodingKeys: String, CodingKey {
        case ok, status, stage, progress, message, error, hits
        case runID = "run_id"
        case processedFrames = "processed_frames"
        case totalFrames = "total_frames"
    }
}

struct LatestCalibration: Equatable {
    let runID: String
    /// Raw JSON re-posted verbatim as /api/track's calibration_json field.
    let calibrationJSON: String

    init(responseData: Data) throws {
        guard let object = try JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let runID = object["run_id"] as? String,
              let calibration = object["calibration"],
              JSONSerialization.isValidJSONObject(calibration) else {
            throw APIError.badResponse
        }
        self.runID = runID
        let data = try JSONSerialization.data(withJSONObject: calibration)
        self.calibrationJSON = String(decoding: data, as: UTF8.self)
    }
}

enum APIError: LocalizedError, Equatable {
    case badResponse
    case http(Int, String?)
    case noCalibration

    var errorDescription: String? {
        switch self {
        case .badResponse: return "The server sent an unexpected response."
        case .http(let code, let message): return message ?? "Server error (\(code))."
        case .noCalibration:
            return "No court calibration found. Calibrate one run from the web app first."
        }
    }
}
