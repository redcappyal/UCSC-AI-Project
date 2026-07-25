import Foundation

protocol APIClientProtocol: Sendable {
    func latestCalibration() async throws -> LatestCalibration
    func upload(videoURL: URL) async throws -> UploadResponse
    func startTrack(videoID: String, calibrationJSON: String, duration: Double,
                    sessionID: String?, cameraRole: String?,
                    peerVideoID: String?, syncManifestJSON: String?) async throws -> JobStatus
    func trackStatus(runID: String) async throws -> JobStatus
    func fetchSolvedCameraModel(calibrationJSON: String) async throws -> String
}

extension APIClientProtocol {
    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus {
        try await startTrack(videoID: videoID, calibrationJSON: calibrationJSON,
                             duration: duration, sessionID: nil, cameraRole: nil,
                             peerVideoID: nil, syncManifestJSON: nil)
    }
}

struct APIClient: APIClientProtocol {
    var baseURL: URL = Config.baseURL
    var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 15 * 60   // big rally uploads
        return URLSession(configuration: config)
    }()

    func latestCalibration() async throws -> LatestCalibration {
        let url = baseURL.appending(path: "api/calibration/latest")
        let (data, response) = try await session.data(from: url)
        if (response as? HTTPURLResponse)?.statusCode == 404 { throw APIError.noCalibration }
        try Self.checkHTTP(response, data: data)
        return try LatestCalibration(responseData: data)
    }

    func upload(videoURL: URL) async throws -> UploadResponse {
        let boundary = "slc-\(UUID().uuidString)"
        var request = URLRequest(url: baseURL.appending(path: "api/upload"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")
        let bodyURL = try Multipart.writeBody(
            boundary: boundary, fields: [],
            fileField: "video_file", filename: videoURL.lastPathComponent,
            contentType: "video/mp4", fileURL: videoURL)
        defer { try? FileManager.default.removeItem(at: bodyURL) }
        let (data, response) = try await session.upload(for: request, fromFile: bodyURL)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(UploadResponse.self, from: data)
    }

    func startTrack(videoID: String, calibrationJSON: String, duration: Double,
                    sessionID: String? = nil, cameraRole: String? = nil,
                    peerVideoID: String? = nil,
                    syncManifestJSON: String? = nil) async throws -> JobStatus {
        var request = URLRequest(url: baseURL.appending(path: "api/track"))
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded",
                         forHTTPHeaderField: "Content-Type")
        var fields: [(String, String)] = [
            ("video_id", videoID),
            ("calibration_json", calibrationJSON),
            ("start_time", "0"),
            ("end_time", String(duration)),
            ("frame_stride", "4"),
            ("inference_width", "960"),
            ("event_engine", ""),
            ("fusion_3d", ""),
        ]
        // The server treats these as strictly optional and requires session and
        // role together — omitting them entirely must stay byte-identical to
        // the single-camera path, so append only what is actually set.
        if let sessionID, let cameraRole {
            fields.append(("session_id", sessionID))
            fields.append(("camera_role", cameraRole))
        }
        if let peerVideoID { fields.append(("peer_video_id", peerVideoID)) }
        if let syncManifestJSON { fields.append(("sync_manifest_json", syncManifestJSON)) }
        request.httpBody = Data(Multipart.formURLEncoded(fields).utf8)
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    func trackStatus(runID: String) async throws -> JobStatus {
        let url = baseURL.appending(path: "api/track/status/\(runID)")
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    /// Phase 3: solves the calibration server-side and hands back the
    /// camera model as JSON. As of Phase 3 this is what phones exchange over
    /// the peer layer's `.calibration` message — the SOLVED model, not raw
    /// wizard taps.
    func fetchSolvedCameraModel(calibrationJSON: String) async throws -> String {
        var request = URLRequest(url: baseURL.appending(path: "api/camera-model"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(
            withJSONObject: ["calibration_json": calibrationJSON])
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["status"] as? String == "ok",
              let cameraModel = object["camera_model"],
              JSONSerialization.isValidJSONObject(cameraModel) else {
            throw APIError.badResponse
        }
        let modelData = try JSONSerialization.data(withJSONObject: cameraModel)
        return String(decoding: modelData, as: UTF8.self)
    }

    private struct ErrorBody: Decodable {
        let error: String?
    }

    private static func checkHTTP(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard (200..<300).contains(http.statusCode) else {
            let message = (try? JSONDecoder().decode(ErrorBody.self, from: data))?.error
            throw APIError.http(http.statusCode, message)
        }
    }
}
